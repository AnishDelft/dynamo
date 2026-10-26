// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Experimental hint-policy plugins: request-scoped worker-selection advice and KV actions.
//!
//! A [`HintPolicy`] maps request signals and serving state to router decisions, engine-directed
//! KV actions, or both. One registered policy packages two optional stages:
//!
//! - **Selection stage** ([`HintPolicy::advise_selection`]): runs on the scheduler actor after the
//!   configured worker-selection filters, before its scorers and picker. It may exclude candidates,
//!   add cost contributions, and attach a per-candidate plan. A plan travels to the KV-action stage
//!   only when its candidate wins.
//! - **KV-action stage** ([`HintPolicy::formulate`]): runs once per booked selection, after booking
//!   and outside the scheduler actor. It returns [`KvHintAction`]s for the selected worker.
//!
//! The router composes its own fetch, if any, and then every policy's actions in configured order
//! into one [`crate::kv_hints::KvHint`]. Hosts own eligibility, booking, and delivery. Policies
//! never reserve capacity, and the engine may still ignore or miss any action.
//!
//! This module is experimental: names and signatures can change without a deprecation period.

mod config;
mod registry;
pub(crate) mod runtime;
#[cfg(test)]
mod tests;

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use crate::kv_hints::KvHintAction;
use crate::plugins::worker_selection::{WorkerCandidates, WorkerInputs, WorkerSelectionContext};
use crate::protocols::WorkerWithDpRank;

pub(crate) use config::RawHintPolicyConfig;
pub use config::{HintPolicyConfig, HintPolicyInstanceConfig};
pub(crate) use registry::HintPolicyRegistry;
pub use registry::{
    HintPolicyConstructor, HintPolicyDefinition, HintPolicyFactory, HintPolicyParameters,
    HintPolicyProvider, HintPolicyProviderError, HintPolicyRegistryError,
};
pub(crate) use runtime::{ActionStage, SelectionStage};
use runtime::{RowEffects, TypedActionStage, TypedSelectionStage};

/// A statically linked hint policy. One instance serves one routing partition.
///
/// Both stages are optional and default to doing nothing; [`Self::requirements`] declares which
/// ones the host calls. Callbacks are synchronous, run on shared router tasks, and must be bounded:
/// no I/O, awaits, or unbounded index traversal.
pub trait HintPolicy: Send + Sync + 'static {
    /// Data a candidate's selection advice carries to the KV-action stage when that candidate
    /// wins. Use `()` for a policy without candidate plans.
    type Plan: Send + 'static;

    /// The stages and inputs this policy uses. Read once when the instance is constructed.
    fn requirements(&self) -> HintRequirements;

    /// Advise worker selection for one evaluation of a request.
    ///
    /// Called on the scheduler actor every time the request is evaluated, including previews
    /// that never dispatch and re-evaluations while it waits in a queue. Advice must be
    /// repeatable and side-effect free: the host discards it unless this evaluation books.
    /// `candidates` are the workers that survived host eligibility and policy filters, restricted
    /// to the inputs declared through [`HintRequirements::with_selection`].
    fn advise_selection(
        &self,
        _context: &HintSelectionContext<'_>,
        _candidates: WorkerCandidates<'_>,
        _advice: &mut SelectionAdvice<'_, Self::Plan>,
    ) -> Result<(), HintPolicyError> {
        Ok(())
    }

    /// Formulate KV actions for the selected worker after booking.
    ///
    /// `plan` is this policy's plan for the selected candidate, if its selection stage attached
    /// one. An error with a plan fails the selection and releases its booking; an error without
    /// one drops only this policy's actions.
    fn formulate(
        &self,
        _context: &HintActionContext<'_>,
        _plan: Option<Self::Plan>,
    ) -> Result<Vec<KvHintAction>, HintPolicyError> {
        Ok(Vec::new())
    }
}

/// The stages a hint policy runs and the inputs each one reads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HintRequirements {
    selection: Option<WorkerInputs>,
    actions: bool,
}

impl HintRequirements {
    /// No stage and no inputs: a policy that the host never calls.
    pub const fn new() -> Self {
        Self {
            selection: None,
            actions: false,
        }
    }

    /// Run the selection stage with these worker input groups.
    pub fn with_selection(mut self, inputs: WorkerInputs) -> Self {
        self.selection = Some(inputs);
        self
    }

    /// Run the KV-action stage for booked selections.
    pub fn with_actions(mut self) -> Self {
        self.actions = true;
        self
    }

    pub fn selection_inputs(&self) -> Option<WorkerInputs> {
        self.selection
    }

    pub fn uses_actions(&self) -> bool {
        self.actions
    }

    /// Requirements a host must satisfy to run every policy in a list.
    pub(crate) fn union(self, other: Self) -> Self {
        Self {
            selection: match (self.selection, other.selection) {
                (Some(left), Some(right)) => Some(left | right),
                (left, right) => left.or(right),
            },
            actions: self.actions || other.actions,
        }
    }

    /// Worker inputs a host must provide for these requirements. Hosts union this with the
    /// worker-selection policy's inputs before starting indexes and subscriptions.
    pub fn worker_inputs(&self) -> WorkerInputs {
        self.selection.unwrap_or(WorkerInputs::NONE)
    }
}

/// Request-level values available to a selection stage.
pub struct HintSelectionContext<'a> {
    request: &'a WorkerSelectionContext<'a>,
}

impl<'a> HintSelectionContext<'a> {
    /// The request metadata the configured worker-selection components also read.
    pub fn request(&self) -> &WorkerSelectionContext<'a> {
        self.request
    }
}

/// Write-only advice from one policy for one evaluation. Rows are the candidate rows passed to
/// [`HintPolicy::advise_selection`].
pub struct SelectionAdvice<'a, P> {
    effects: &'a mut RowEffects,
    plans: &'a mut [Option<P>],
}

impl<P> SelectionAdvice<'_, P> {
    /// Number of candidate rows.
    pub fn len(&self) -> usize {
        self.plans.len()
    }

    pub fn is_empty(&self) -> bool {
        self.plans.is_empty()
    }

    /// Remove a candidate from this evaluation. Exclusions from every policy apply together; a
    /// policy cannot restore a worker that the host or another component excluded.
    pub fn exclude(&mut self, row: usize) -> Result<(), HintPolicyError> {
        self.check_row(row)?;
        self.effects.exclude(row);
        Ok(())
    }

    /// Add a finite, lower-is-better contribution to a candidate's total cost, in the configured
    /// worker-selection policy's units. Contributions from several calls and policies add up, and
    /// add to what the configured scorers already charge or credit, so do not credit reuse, such
    /// as cached or fetchable blocks, that those scorers already credit.
    pub fn add_cost(&mut self, row: usize, cost: f64) -> Result<(), HintPolicyError> {
        self.check_row(row)?;
        if !cost.is_finite() {
            return Err(HintPolicyError::new(format!(
                "non-finite cost {cost} for candidate row {row}"
            )));
        }
        self.effects.add_cost(row, cost);
        Ok(())
    }

    /// Attach this policy's plan to a candidate, replacing an earlier one. The plan reaches
    /// [`HintPolicy::formulate`] only if this candidate wins this evaluation and books.
    pub fn plan(&mut self, row: usize, plan: P) -> Result<(), HintPolicyError> {
        self.check_row(row)?;
        self.plans[row] = Some(plan);
        Ok(())
    }

    fn check_row(&self, row: usize) -> Result<(), HintPolicyError> {
        if row < self.plans.len() {
            Ok(())
        } else {
            Err(HintPolicyError::new(format!(
                "candidate row {row} is out of range for {} rows",
                self.plans.len()
            )))
        }
    }
}

/// Values available to the KV-action stage for one booked selection.
pub struct HintActionContext<'a> {
    request_id: &'a str,
    worker: WorkerWithDpRank,
    prompt_tokens: usize,
    block_size: u32,
}

impl HintActionContext<'_> {
    /// The routing request ID, which the composed hint also uses as its message ID.
    pub fn request_id(&self) -> &str {
        self.request_id
    }

    /// The booked worker and rank every action targets.
    pub fn worker(&self) -> WorkerWithDpRank {
        self.worker
    }

    /// Exact incoming prompt length in tokens.
    pub fn prompt_tokens(&self) -> usize {
        self.prompt_tokens
    }

    /// Tokens per KV block in this partition.
    pub fn block_size(&self) -> u32 {
        self.block_size
    }
}

/// A policy failure reported to the router. The message is included in logs and errors.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct HintPolicyError {
    message: String,
}

impl HintPolicyError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// One named policy instance for one routing partition, created by a provider's factory.
pub struct HintPolicyInstance {
    name: Arc<str>,
    requirements: HintRequirements,
    selection: Option<Box<dyn SelectionStage>>,
    actions: Option<Arc<dyn ActionStage>>,
}

impl HintPolicyInstance {
    /// Wrap a policy. The instance shares the policy between its selection and KV-action stages.
    pub fn new<P: HintPolicy>(name: impl Into<Arc<str>>, policy: P) -> Self {
        let name = name.into();
        let requirements = policy.requirements();
        let policy = Arc::new(policy);
        Self {
            selection: requirements.selection.map(|inputs| {
                Box::new(TypedSelectionStage::new(
                    name.clone(),
                    Arc::clone(&policy),
                    inputs,
                )) as Box<dyn SelectionStage>
            }),
            // Plans always need a KV-action stage, even a policy whose `formulate` does nothing.
            actions: (requirements.actions || requirements.selection.is_some())
                .then(|| Arc::new(TypedActionStage::new(policy)) as Arc<dyn ActionStage>),
            name,
            requirements,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn requirements(&self) -> HintRequirements {
        self.requirements
    }

    pub(crate) fn into_stages(self) -> HintPolicyStages {
        HintPolicyStages {
            name: self.name,
            selection: self.selection,
            actions: self.actions,
        }
    }
}

/// One instance's stages, split for the scheduler actor and the selection core.
pub(crate) struct HintPolicyStages {
    pub(crate) name: Arc<str>,
    pub(crate) selection: Option<Box<dyn SelectionStage>>,
    pub(crate) actions: Option<Arc<dyn ActionStage>>,
}

impl fmt::Debug for HintPolicyInstance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HintPolicyInstance")
            .field("name", &self.name)
            .field("requirements", &self.requirements)
            .finish_non_exhaustive()
    }
}

/// The winning candidate's hint plans from one evaluation, bound to that worker and rank.
///
/// Produced by the scheduler for the selected worker and consumed when the booking's KV hint is
/// formulated. Losing candidates' plans are dropped on the scheduler actor.
#[derive(Default)]
pub struct HintPlans {
    worker: Option<WorkerWithDpRank>,
    plans: Vec<(usize, Box<dyn Any + Send>)>,
}

impl HintPlans {
    pub(crate) fn new(worker: WorkerWithDpRank, plans: Vec<(usize, Box<dyn Any + Send>)>) -> Self {
        Self {
            worker: Some(worker),
            plans,
        }
    }

    /// Whether no policy attached a plan to the selected worker.
    pub fn is_empty(&self) -> bool {
        self.plans.is_empty()
    }

    pub(crate) fn worker(&self) -> Option<WorkerWithDpRank> {
        self.worker
    }

    pub(crate) fn take_plan(&mut self, policy: usize) -> Option<Box<dyn Any + Send>> {
        let index = self.plans.iter().position(|(owner, _)| *owner == policy)?;
        Some(self.plans.swap_remove(index).1)
    }
}

impl fmt::Debug for HintPlans {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HintPlans")
            .field("worker", &self.worker)
            .field(
                "plan_policies",
                &self
                    .plans
                    .iter()
                    .map(|(policy, _)| policy)
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}
