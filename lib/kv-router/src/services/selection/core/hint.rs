// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The KV transfer hint attached to a booked selection, and composition of a booked selection's
//! KV hint from the router's fetch and its hint policies.

use std::collections::HashSet;

use super::*;
use crate::kv_hints::KvTransferSourceChoice;
use crate::plugins::hint_policy::HintPlans;
use crate::plugins::hint_policy::runtime::{ActionInputs, ActionStage};

/// Action ID of the router's `kv.fetch` action. No other action may use it.
pub(super) const KV_HINT_FETCH_ACTION_ID: &str = "a1";
/// Upper bound on the actions one composed hint carries.
const MAX_KV_HINT_ACTIONS: usize = 32;
const KV_FETCH_ACTION_TYPE: &str = "kv.fetch";

/// Whether any worker in the published partition snapshot advertises a router
/// hint worker type (worker-level metadata, so one rank suffices).
pub(super) fn hint_capable_partition(configs: &HashMap<WorkerId, SelectionWorkerConfig>) -> bool {
    configs.values().any(|config| {
        config
            .router_hint_worker_type
            .as_deref()
            .is_some_and(|worker_type| !worker_type.is_empty())
    })
}

/// The fetch payload for the source worker selection chose (see
/// [`crate::kv_hints::KvTransferSourceSummary`]). The endpoint comes from the
/// current snapshot, so a source that left since selection gets no hint.
pub(super) fn transfer_hint_payload(
    configs: &HashMap<WorkerId, SelectionWorkerConfig>,
    source: KvTransferSourceChoice,
    candidates: &KvTransferCandidates,
) -> Option<KvSourceLocationsPayload> {
    let source_control_endpoint = match source.source {
        KvTransferCandidateSource::Worker(worker) => configs
            .get(&worker.worker_id)?
            .kv_hint_transfer_metadata_for_dp_rank(worker.dp_rank)?
            .source_control_endpoint?
            .to_string(),
        KvTransferCandidateSource::CacheOwner(owner) => candidates
            .routing_snapshot
            .as_ref()?
            .router_hint_source(owner)?
            .metadata
            .source_control_endpoint
            .clone(),
    };
    let block_hashes = candidates.block_hashes.get(..source.prefix_blocks)?;
    if source_control_endpoint.is_empty() || block_hashes.is_empty() {
        return None;
    }
    Some(KvSourceLocationsPayload {
        source_control_endpoint,
        block_hashes: block_hashes.to_vec(),
    })
}

/// One configured hint policy of a partition, in configured order.
pub(super) struct HintPolicySlot {
    pub(super) name: Arc<str>,
    /// `None` for a policy without a KV-action stage.
    pub(super) actions: Option<Arc<dyn ActionStage>>,
}

/// Formulate one booking's KV hint: the router's fetch, if any, then its partition's hint
/// policies' actions in configured order.
///
/// A policy's output is accepted or rejected whole. Output is rejected when `formulate` fails, an
/// action ID repeats, an action other than the router's `kv.fetch` uses its ID, a second
/// `kv.fetch` appears, or the hint would exceed [`MAX_KV_HINT_ACTIONS`]. The output of policies
/// that committed a plan for the selected worker is checked first, so no other policy's output can
/// crowd it out; rejecting it fails the booking, because the plan may have decided worker
/// selection.
/// Rejecting any other output drops only that policy's actions.
pub(super) fn formulate_kv_hint(
    policies: &[HintPolicySlot],
    mut plans: HintPlans,
    inputs: &ActionInputs<'_>,
    router_fetch: Option<KvHintAction>,
) -> Result<Option<KvHint>, SelectionError> {
    if plans.worker().is_some_and(|worker| worker != inputs.worker) {
        return Err(SelectionError::Internal(format!(
            "hint plans were made for worker {:?}, but the request booked {:?}",
            plans.worker(),
            inputs.worker
        )));
    }
    let mut outputs = Vec::with_capacity(policies.len());
    for (index, policy) in policies.iter().enumerate() {
        let plan = plans.take_plan(index);
        let committed = plan.is_some();
        let Some(stage) = &policy.actions else {
            continue;
        };
        match stage.formulate(&inputs.context(), plan) {
            Ok(actions) if !actions.is_empty() => outputs.push(PolicyOutput {
                policy,
                committed,
                actions,
                accepted: false,
            }),
            Ok(_) => {}
            Err(error) => reject(policy, committed, inputs, &error.to_string())?,
        }
    }

    let mut composer = Composer::new(router_fetch.as_ref());
    for committed in [true, false] {
        for output in outputs
            .iter_mut()
            .filter(|output| output.committed == committed)
        {
            match composer.admit(&output.actions) {
                Ok(()) => output.accepted = true,
                Err(reason) => reject(output.policy, committed, inputs, &reason)?,
            }
        }
    }
    let actions: Vec<KvHintAction> = router_fetch
        .into_iter()
        .chain(
            outputs
                .into_iter()
                .filter(|output| output.accepted)
                .flat_map(|output| output.actions),
        )
        .collect();
    Ok((!actions.is_empty()).then(|| KvHint::new(inputs.request_id, actions)))
}

/// One policy's formulated actions, waiting for composition.
struct PolicyOutput<'a> {
    policy: &'a HintPolicySlot,
    /// Whether the policy committed a plan for the selected worker.
    committed: bool,
    actions: Vec<KvHintAction>,
    accepted: bool,
}

/// The action IDs, fetch, and size of the hint composed so far.
struct Composer {
    action_ids: HashSet<String>,
    has_fetch: bool,
    len: usize,
}

impl Composer {
    fn new(router_fetch: Option<&KvHintAction>) -> Self {
        Self {
            action_ids: router_fetch
                .map(|fetch| fetch.action_id.clone())
                .into_iter()
                .collect(),
            has_fetch: router_fetch.is_some(),
            len: usize::from(router_fetch.is_some()),
        }
    }

    /// Add one policy's output if it composes, or explain why it does not.
    fn admit(&mut self, output: &[KvHintAction]) -> Result<(), String> {
        let mut new_ids = HashSet::with_capacity(output.len());
        let mut fetches = 0;
        for action in output {
            let is_fetch = action.action_type == KV_FETCH_ACTION_TYPE;
            if action.action_id == KV_HINT_FETCH_ACTION_ID && !is_fetch {
                return Err(format!(
                    "action ID {KV_HINT_FETCH_ACTION_ID:?} is reserved for the router's kv.fetch"
                ));
            }
            if self.action_ids.contains(&action.action_id)
                || !new_ids.insert(action.action_id.as_str())
            {
                return Err(format!("action ID {:?} is already used", action.action_id));
            }
            fetches += usize::from(is_fetch);
        }
        if fetches > usize::from(!self.has_fetch) {
            return Err("a hint carries at most one kv.fetch".to_string());
        }
        if self.len + output.len() > MAX_KV_HINT_ACTIONS {
            return Err(format!(
                "a hint carries at most {MAX_KV_HINT_ACTIONS} actions"
            ));
        }
        self.has_fetch |= fetches > 0;
        self.len += output.len();
        self.action_ids
            .extend(output.iter().map(|action| action.action_id.clone()));
        Ok(())
    }
}

fn reject(
    policy: &HintPolicySlot,
    committed: bool,
    inputs: &ActionInputs<'_>,
    reason: &str,
) -> Result<(), SelectionError> {
    if committed {
        return Err(SelectionError::Internal(format!(
            "hint policy {:?} could not deliver its plan for the selected worker: {reason}",
            policy.name
        )));
    }
    tracing::warn!(
        request_id = inputs.request_id,
        hint_policy = %policy.name,
        reason,
        "Dropping hint-policy actions"
    );
    Ok(())
}
