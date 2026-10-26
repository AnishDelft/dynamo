// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host-side execution of hint-policy stages.
//!
//! The scheduler actor runs selection stages through [`HintSelection`], which owns each policy's
//! per-evaluation plan buffer and the effects shared by all policies. Only the winning row's plans
//! leave the actor, as [`HintPlans`]. The selection core runs KV-action stages through
//! [`ActionStage`] after booking.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use super::{
    HintActionContext, HintPlans, HintPolicy, HintPolicyError, HintRequirements,
    HintSelectionContext, KvFetchSource, SelectionAdvice,
};
use crate::kv_hints::{
    KvHintAction, KvSourceLocationsPayload, KvTransferCandidateSource, KvTransferCandidates,
};
use crate::plugins::worker_selection::{
    CacheSnapshot, CandidateData, WorkerCandidates, WorkerInputs, WorkerSelectionContext,
};
use crate::protocols::WorkerWithDpRank;
use crate::scheduling::{KvSchedulerError, TierOverlapBlocks, WorkerSelectionPolicyError};

/// Action ID of the router's `kv.fetch` action. No other action may use it, so a composed hint
/// carries at most one fetch.
pub(crate) const KV_HINT_FETCH_ACTION_ID: &str = "a1";

/// The host's router-hint source rules over one partition's worker snapshot.
pub(crate) trait FetchSourceRules: Send + Sync {
    /// The best eligible source for `target` holding more than `local_prefix_blocks`, and the
    /// prefix it holds in blocks.
    fn source_for(
        &self,
        target: WorkerWithDpRank,
        local_prefix_blocks: usize,
        candidates: &KvTransferCandidates,
    ) -> Option<(KvTransferCandidateSource, usize)>;

    /// The `kv.fetch` payload for `source`'s first `prefix_blocks` blocks, or `None` when the
    /// source no longer has a control endpoint.
    fn payload(
        &self,
        source: KvTransferCandidateSource,
        prefix_blocks: usize,
        candidates: &KvTransferCandidates,
    ) -> Option<KvSourceLocationsPayload>;
}

/// Request-scoped inputs the selection core prepares for hint policies outside the scheduler
/// actor. Default when the request has no hint inputs.
#[derive(Clone, Default)]
pub struct HintRequestInputs {
    fetch_rules: Option<Arc<dyn FetchSourceRules>>,
}

impl HintRequestInputs {
    pub(crate) fn with_fetch_rules(rules: Arc<dyn FetchSourceRules>) -> Self {
        Self {
            fetch_rules: Some(rules),
        }
    }

    pub(crate) fn fetch_rules(&self) -> Option<&dyn FetchSourceRules> {
        self.fetch_rules.as_deref()
    }
}

impl fmt::Debug for HintRequestInputs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HintRequestInputs")
            .field("fetch_rules", &self.fetch_rules.is_some())
            .finish()
    }
}

/// Where a fetch target's local prefix comes from.
#[derive(Clone, Copy)]
enum LocalPrefix<'a> {
    /// Every candidate's device and host-pinned overlap, during selection.
    Tiers(&'a TierOverlapBlocks),
    /// The booked worker's prefix, after selection.
    Selected(WorkerWithDpRank, usize),
}

/// One request's fetch sources: the host rules, its retained block chain, and local prefixes.
#[derive(Clone, Copy)]
pub(crate) struct FetchView<'a> {
    rules: &'a dyn FetchSourceRules,
    candidates: &'a KvTransferCandidates,
    local: LocalPrefix<'a>,
}

impl<'a> FetchView<'a> {
    pub(crate) fn for_selection(
        rules: &'a dyn FetchSourceRules,
        candidates: &'a KvTransferCandidates,
        tiers: &'a TierOverlapBlocks,
    ) -> Self {
        Self {
            rules,
            candidates,
            local: LocalPrefix::Tiers(tiers),
        }
    }

    pub(crate) fn for_selected(
        rules: &'a dyn FetchSourceRules,
        candidates: &'a KvTransferCandidates,
        worker: WorkerWithDpRank,
        local_prefix_blocks: usize,
    ) -> Self {
        Self {
            rules,
            candidates,
            local: LocalPrefix::Selected(worker, local_prefix_blocks),
        }
    }

    fn local_prefix_blocks(&self, target: WorkerWithDpRank) -> Option<usize> {
        match self.local {
            LocalPrefix::Tiers(tiers) => Some(tiers.fetch_local_prefix_blocks(target)),
            LocalPrefix::Selected(worker, blocks) => (worker == target).then_some(blocks),
        }
    }

    pub(crate) fn source_for(&self, target: WorkerWithDpRank) -> Option<KvFetchSource> {
        let local = self.local_prefix_blocks(target)?;
        let (source, prefix_blocks) = self.rules.source_for(target, local, self.candidates)?;
        (prefix_blocks > local).then(|| KvFetchSource::new(target, source, prefix_blocks, local))
    }

    pub(crate) fn action(&self, source: &KvFetchSource) -> Option<KvHintAction> {
        let payload = self
            .rules
            .payload(source.source, source.prefix_blocks, self.candidates)?;
        Some(KvHintAction::fetch(KV_HINT_FETCH_ACTION_ID, payload))
    }
}

/// Effects every policy writes for one evaluation, index-aligned with its candidate rows.
#[derive(Default)]
pub(crate) struct RowEffects {
    workers: Vec<WorkerWithDpRank>,
    keep: Vec<bool>,
    any_excluded: bool,
    costs: Vec<f64>,
    fetches: Vec<Option<(usize, KvFetchSource)>>,
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
        self.fetches.clear();
        self.fetches.resize(rows.len(), None);
    }

    pub(super) fn exclude(&mut self, row: usize) {
        self.keep[row] = false;
        self.any_excluded = true;
    }

    pub(super) fn add_cost(&mut self, row: usize, cost: f64) {
        self.costs[row] += cost;
    }

    pub(super) fn fetch(
        &mut self,
        policy: usize,
        row: usize,
        source: KvFetchSource,
    ) -> Result<(), HintPolicyError> {
        if source.target != self.workers[row] {
            return Err(HintPolicyError::new(format!(
                "fetch source for worker {:?} was attached to candidate row {row} ({:?})",
                source.target, self.workers[row]
            )));
        }
        if self.fetches[row].is_some() {
            return Err(HintPolicyError::new(format!(
                "candidate row {row} already has a fetch"
            )));
        }
        self.fetches[row] = Some((policy, source));
        Ok(())
    }

    fn retain(&mut self) {
        let mut rows = self.keep.iter().copied();
        self.workers.retain(|_| rows.next().unwrap_or(false));
        let mut rows = self.keep.iter().copied();
        self.costs.retain(|_| rows.next().unwrap_or(false));
        let mut rows = self.keep.iter().copied();
        self.fetches.retain(|_| rows.next().unwrap_or(false));
        self.keep.retain(|keep| *keep);
        self.any_excluded = false;
    }
}

/// A policy's selection stage, erased so one actor can run policies with different plan types.
pub(crate) trait SelectionStage: Send {
    fn name(&self) -> &str;
    fn inputs(&self) -> WorkerInputs;
    fn reads_kv_fetch_sources(&self) -> bool;
    fn advise(
        &mut self,
        policy: usize,
        request: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        fetch: Option<FetchView<'_>>,
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
    reads_fetch: bool,
    plans: Vec<Option<P::Plan>>,
}

impl<P: HintPolicy> TypedSelectionStage<P> {
    pub(super) fn new(
        name: Arc<str>,
        policy: Arc<P>,
        inputs: WorkerInputs,
        reads_fetch: bool,
    ) -> Self {
        Self {
            name,
            policy,
            inputs,
            reads_fetch,
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

    fn reads_kv_fetch_sources(&self) -> bool {
        self.reads_fetch
    }

    fn advise(
        &mut self,
        policy: usize,
        request: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        fetch: Option<FetchView<'_>>,
        effects: &mut RowEffects,
    ) -> Result<(), HintPolicyError> {
        self.plans.clear();
        self.plans.resize_with(candidates.len(), || None);
        let fetch = fetch.filter(|_| self.reads_fetch);
        let context = HintSelectionContext { request, fetch };
        let mut advice = SelectionAdvice {
            policy,
            effects,
            plans: &mut self.plans,
            fetch,
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
    reads_fetch: bool,
    effects: RowEffects,
}

impl HintSelection {
    pub(crate) fn new(stages: Vec<(usize, Box<dyn SelectionStage>)>) -> Self {
        let inputs = stages
            .iter()
            .fold(WorkerInputs::NONE, |inputs, (_, stage)| {
                inputs | stage.inputs()
            });
        let reads_fetch = stages
            .iter()
            .any(|(_, stage)| stage.reads_kv_fetch_sources());
        Self {
            stages,
            inputs,
            reads_fetch,
            effects: RowEffects::default(),
        }
    }

    /// The union of the worker inputs the stages read from candidate rows.
    pub(crate) fn inputs(&self) -> WorkerInputs {
        self.inputs
    }

    pub(crate) fn reads_kv_fetch_sources(&self) -> bool {
        self.reads_fetch
    }

    /// Run every stage on `rows`. Effects stay pending until [`Self::retain_kept`] and
    /// [`Self::costs`] apply them.
    pub(crate) fn advise(
        &mut self,
        request: &WorkerSelectionContext<'_>,
        rows: &[CandidateData],
        cache_snapshot: &CacheSnapshot<'_>,
        fetch: Option<FetchView<'_>>,
    ) -> Result<(), KvSchedulerError> {
        self.effects.reset(rows);
        for (policy, stage) in &mut self.stages {
            stage
                .advise(
                    *policy,
                    request,
                    WorkerCandidates::new(rows, stage.inputs(), cache_snapshot),
                    fetch,
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

    /// The selected row's plans and fetch. Every other row's plans are dropped here.
    pub(crate) fn take_winner(&mut self, row: usize) -> HintPlans {
        let worker = self.effects.workers[row];
        let plans = self
            .stages
            .iter_mut()
            .filter_map(|(policy, stage)| Some((*policy, stage.take_plan(row)?)))
            .collect();
        HintPlans::new(worker, plans, self.effects.fetches[row])
    }
}

/// A policy's KV-action stage, erased so the selection core can run a list of policies.
pub(crate) trait ActionStage: Send + Sync {
    fn requirements(&self) -> HintRequirements;
    fn formulate(
        &self,
        context: &HintActionContext<'_>,
        plan: Option<Box<dyn Any + Send>>,
    ) -> Result<Vec<KvHintAction>, HintPolicyError>;
}

pub(super) struct TypedActionStage<P: HintPolicy> {
    policy: Arc<P>,
    requirements: HintRequirements,
}

impl<P: HintPolicy> TypedActionStage<P> {
    pub(super) fn new(policy: Arc<P>, requirements: HintRequirements) -> Self {
        Self {
            policy,
            requirements,
        }
    }
}

impl<P: HintPolicy> ActionStage for TypedActionStage<P> {
    fn requirements(&self) -> HintRequirements {
        self.requirements
    }

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
    pub(crate) fetch: Option<FetchView<'a>>,
}

impl<'a> ActionInputs<'a> {
    /// The context one stage sees, restricted to its declared inputs.
    pub(crate) fn context(
        &self,
        requirements: HintRequirements,
        fetch_taken: bool,
    ) -> HintActionContext<'a> {
        HintActionContext {
            request_id: self.request_id,
            worker: self.worker,
            prompt_tokens: self.prompt_tokens,
            block_size: self.block_size,
            fetch: self.fetch.filter(|_| requirements.reads_kv_fetch_sources()),
            fetch_taken,
        }
    }
}
