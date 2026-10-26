// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host-side execution of hint-policy stages.
//!
//! The scheduler actor runs selection stages through [`HintSelection`], which owns each policy's
//! per-evaluation plan buffer and the effects shared by all policies. Only the winning row's plans
//! leave the actor, as [`HintPlans`]. The selection core runs KV-action stages through
//! [`ActionStage`] after booking.

use std::any::Any;
use std::sync::Arc;

use super::{
    HintActionContext, HintPlans, HintPolicy, HintPolicyError, HintSelectionContext,
    SelectionAdvice,
};
use crate::kv_hints::KvHintAction;
use crate::plugins::worker_selection::{
    CacheSnapshot, CandidateData, WorkerCandidates, WorkerInputs, WorkerSelectionContext,
};
use crate::protocols::WorkerWithDpRank;
use crate::scheduling::{KvSchedulerError, WorkerSelectionPolicyError};

/// Effects every policy writes for one evaluation, index-aligned with its candidate rows.
#[derive(Default)]
pub(crate) struct RowEffects {
    workers: Vec<WorkerWithDpRank>,
    keep: Vec<bool>,
    any_excluded: bool,
    costs: Vec<f64>,
}

impl RowEffects {
    fn reset(&mut self, rows: &[CandidateData]) {
        self.workers.clear();
        self.workers.extend(rows.iter().map(|row| row.worker));
        self.keep.clear();
        self.keep.resize(rows.len(), true);
        self.any_excluded = false;
        self.costs.clear();
        self.costs.resize(rows.len(), 0.0);
    }

    pub(super) fn exclude(&mut self, row: usize) {
        self.keep[row] = false;
        self.any_excluded = true;
    }

    pub(super) fn add_cost(&mut self, row: usize, cost: f64) {
        self.costs[row] += cost;
    }

    fn retain(&mut self) {
        let mut rows = self.keep.iter().copied();
        self.workers.retain(|_| rows.next().unwrap_or(false));
        let mut rows = self.keep.iter().copied();
        self.costs.retain(|_| rows.next().unwrap_or(false));
        self.keep.retain(|keep| *keep);
        self.any_excluded = false;
    }
}

/// A policy's selection stage, erased so one actor can run policies with different plan types.
pub(crate) trait SelectionStage: Send {
    fn name(&self) -> &str;
    fn inputs(&self) -> WorkerInputs;
    fn advise(
        &mut self,
        request: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        effects: &mut RowEffects,
    ) -> Result<(), HintPolicyError>;
    /// Drop the plans of rows that `keep` excludes, preserving order.
    fn retain(&mut self, keep: &[bool]);
    /// Take one row's plan and drop every other row's.
    fn take_plan(&mut self, row: usize) -> Option<Box<dyn Any + Send>>;
}

pub(super) struct TypedSelectionStage<P: HintPolicy> {
    name: Arc<str>,
    policy: Arc<P>,
    inputs: WorkerInputs,
    plans: Vec<Option<P::Plan>>,
}

impl<P: HintPolicy> TypedSelectionStage<P> {
    pub(super) fn new(name: Arc<str>, policy: Arc<P>, inputs: WorkerInputs) -> Self {
        Self {
            name,
            policy,
            inputs,
            plans: Vec::new(),
        }
    }
}

impl<P: HintPolicy> SelectionStage for TypedSelectionStage<P> {
    fn name(&self) -> &str {
        &self.name
    }

    fn inputs(&self) -> WorkerInputs {
        self.inputs
    }

    fn advise(
        &mut self,
        request: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        effects: &mut RowEffects,
    ) -> Result<(), HintPolicyError> {
        self.plans.clear();
        self.plans.resize_with(candidates.len(), || None);
        let context = HintSelectionContext { request };
        let mut advice = SelectionAdvice {
            effects,
            plans: &mut self.plans,
        };
        self.policy
            .advise_selection(&context, candidates, &mut advice)
    }

    fn retain(&mut self, keep: &[bool]) {
        let mut rows = keep.iter().copied();
        self.plans.retain(|_| rows.next().unwrap_or(false));
    }

    fn take_plan(&mut self, row: usize) -> Option<Box<dyn Any + Send>> {
        let plan = self.plans.get_mut(row).and_then(Option::take);
        self.plans.clear();
        plan.map(|plan| Box::new(plan) as Box<dyn Any + Send>)
    }
}

/// The selection stages of one partition's hint policies, run in configured order.
pub(crate) struct HintSelection {
    /// Each stage with its policy's index in the partition's configured list.
    stages: Vec<(usize, Box<dyn SelectionStage>)>,
    inputs: WorkerInputs,
    effects: RowEffects,
}

impl HintSelection {
    pub(crate) fn new(stages: Vec<(usize, Box<dyn SelectionStage>)>) -> Self {
        let inputs = stages
            .iter()
            .fold(WorkerInputs::NONE, |inputs, (_, stage)| {
                inputs | stage.inputs()
            });
        Self {
            stages,
            inputs,
            effects: RowEffects::default(),
        }
    }

    /// The union of the worker inputs the stages read from candidate rows.
    pub(crate) fn inputs(&self) -> WorkerInputs {
        self.inputs
    }

    /// Run every stage on `rows`. Effects stay pending until [`Self::retain_kept`] and
    /// [`Self::costs`] apply them.
    pub(crate) fn advise(
        &mut self,
        request: &WorkerSelectionContext<'_>,
        rows: &[CandidateData],
        cache_snapshot: &CacheSnapshot<'_>,
    ) -> Result<(), KvSchedulerError> {
        self.effects.reset(rows);
        for (_, stage) in &mut self.stages {
            stage
                .advise(
                    request,
                    WorkerCandidates::new(rows, stage.inputs(), cache_snapshot),
                    &mut self.effects,
                )
                .map_err(|error| {
                    WorkerSelectionPolicyError::failed(format!(
                        "hint policy {:?}: {error}",
                        stage.name()
                    ))
                })?;
        }
        Ok(())
    }

    /// The rows to keep when some policy excluded a candidate; `None` when every row stays.
    pub(crate) fn kept_rows(&self) -> Option<&[bool]> {
        self.effects
            .any_excluded
            .then_some(self.effects.keep.as_slice())
    }

    /// Drop excluded rows from the pending effects and every stage's plans.
    pub(crate) fn retain_kept(&mut self) {
        if !self.effects.any_excluded {
            return;
        }
        for (_, stage) in &mut self.stages {
            stage.retain(&self.effects.keep);
        }
        self.effects.retain();
    }

    /// Each kept row's summed cost contribution, index-aligned with the kept rows.
    pub(crate) fn costs(&self) -> &[f64] {
        &self.effects.costs
    }

    /// The selected row's plans. Every other row's plans are dropped here.
    pub(crate) fn take_winner(&mut self, row: usize) -> HintPlans {
        let worker = self.effects.workers[row];
        let plans = self
            .stages
            .iter_mut()
            .filter_map(|(policy, stage)| Some((*policy, stage.take_plan(row)?)))
            .collect();
        HintPlans::new(worker, plans)
    }
}

/// A policy's KV-action stage, erased so the selection core can run a list of policies.
pub(crate) trait ActionStage: Send + Sync {
    fn formulate(
        &self,
        context: &HintActionContext<'_>,
        plan: Option<Box<dyn Any + Send>>,
    ) -> Result<Vec<KvHintAction>, HintPolicyError>;
}

pub(super) struct TypedActionStage<P: HintPolicy> {
    policy: Arc<P>,
}

impl<P: HintPolicy> TypedActionStage<P> {
    pub(super) fn new(policy: Arc<P>) -> Self {
        Self { policy }
    }
}

impl<P: HintPolicy> ActionStage for TypedActionStage<P> {
    fn formulate(
        &self,
        context: &HintActionContext<'_>,
        plan: Option<Box<dyn Any + Send>>,
    ) -> Result<Vec<KvHintAction>, HintPolicyError> {
        let plan = plan
            .map(|plan| {
                plan.downcast::<P::Plan>()
                    .map(|plan| *plan)
                    .map_err(|_| HintPolicyError::new("plan has another policy's type"))
            })
            .transpose()?;
        self.policy.formulate(context, plan)
    }
}

/// Request values the selection core shares with every KV-action stage of one booking.
pub(crate) struct ActionInputs<'a> {
    pub(crate) request_id: &'a str,
    pub(crate) worker: WorkerWithDpRank,
    pub(crate) prompt_tokens: usize,
    pub(crate) block_size: u32,
}

impl<'a> ActionInputs<'a> {
    /// The context every stage sees.
    pub(crate) fn context(&self) -> HintActionContext<'a> {
        HintActionContext {
            request_id: self.request_id,
            worker: self.worker,
            prompt_tokens: self.prompt_tokens,
            block_size: self.block_size,
        }
    }
}
