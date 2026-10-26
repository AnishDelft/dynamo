// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Router-hint fetch rules over a partition's worker snapshot, and composition of a booked
//! selection's KV hint from its hint policies.

use std::collections::HashSet;

use super::*;
use crate::plugins::hint_policy::HintPlans;
use crate::plugins::hint_policy::runtime::{
    ActionInputs, ActionStage, FetchSourceRules, KV_HINT_FETCH_ACTION_ID,
};
use crate::protocols::KvTransferEnforcement;

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

/// The router-hint source rules over one published snapshot of a partition's workers. Rebuilt
/// when the partition publishes a changed worker map, so selections read it without scanning.
pub(super) struct PartitionFetchRules {
    configs: HashMap<WorkerId, SelectionWorkerConfig>,
}

impl PartitionFetchRules {
    /// The rules for `configs`, or `None` when no worker consumes router hints.
    pub(super) fn for_partition(
        configs: &HashMap<WorkerId, SelectionWorkerConfig>,
    ) -> Option<Self> {
        hint_capable_partition(configs).then(|| Self {
            configs: configs.clone(),
        })
    }
}

impl FetchSourceRules for PartitionFetchRules {
    fn source_for(
        &self,
        target: WorkerWithDpRank,
        local_prefix_blocks: usize,
        candidates: &KvTransferCandidates,
    ) -> Option<(KvTransferCandidateSource, usize)> {
        best_fetch_source(&self.configs, target, local_prefix_blocks, candidates)
    }

    fn payload(
        &self,
        source: KvTransferCandidateSource,
        prefix_blocks: usize,
        candidates: &KvTransferCandidates,
    ) -> Option<KvSourceLocationsPayload> {
        fetch_payload(&self.configs, source, prefix_blocks, candidates)
    }
}

/// Pick the best router-hint source for `target`: a same-role worker (or
/// cache owner) holding a longer root-aligned prefix than the target's own
/// `target_cached_prefix_blocks`, with a non-empty control endpoint.
///
/// When the target sets a `kv_transfer_domain`, the source must also be in the
/// target's island: its `topology_domains` value for that domain equals the
/// target's. A source without that value is in no island, and a cache owner
/// that publishes no topology uses its attached worker's. With `preferred`
/// enforcement, a cross-island source is used only when no same-island source
/// qualifies.
fn best_fetch_source(
    configs: &HashMap<WorkerId, SelectionWorkerConfig>,
    target: WorkerWithDpRank,
    target_cached_prefix_blocks: usize,
    candidates: &KvTransferCandidates,
) -> Option<(KvTransferCandidateSource, usize)> {
    let target_config = configs.get(&target.worker_id)?;
    let target_metadata = target_config.kv_hint_transfer_metadata_for_dp_rank(target.dp_rank)?;

    let is_eligible = |source| match source {
        KvTransferCandidateSource::Worker(worker) => {
            worker != target
                && configs.get(&worker.worker_id).is_some_and(|config| {
                    config.kv_event_source_mode.as_deref() != Some("state_agent_v2")
                        && config
                            .kv_hint_transfer_metadata_for_dp_rank(worker.dp_rank)
                            .is_some_and(|source_metadata| {
                                source_metadata.worker_type == target_metadata.worker_type
                                    && source_metadata
                                        .source_control_endpoint
                                        .is_some_and(|endpoint| !endpoint.is_empty())
                            })
                })
        }
        KvTransferCandidateSource::CacheOwner(owner) => candidates
            .routing_snapshot
            .as_ref()
            .and_then(|snapshot| snapshot.router_hint_source(owner))
            .is_some_and(|source| {
                source.attached_worker != Some(target)
                    && source.metadata.worker_type == target_metadata.worker_type
                    && !source.metadata.source_control_endpoint.is_empty()
            }),
    };

    match target_config.kv_transfer_domain.as_deref() {
        None => candidates.best_ranked_source_prefix(target_cached_prefix_blocks, |source| {
            is_eligible(source).then_some(())
        }),
        Some(domain) => {
            let target_island = target_config.topology_domains.get(domain);
            let source_island = |source| match source {
                KvTransferCandidateSource::Worker(worker) => {
                    configs.get(&worker.worker_id)?.topology_domains.get(domain)
                }
                KvTransferCandidateSource::CacheOwner(owner) => {
                    let source = candidates
                        .routing_snapshot
                        .as_ref()?
                        .router_hint_source(owner)?;
                    let topology = if source.metadata.topology_domains.is_empty() {
                        &configs
                            .get(&source.attached_worker?.worker_id)?
                            .topology_domains
                    } else {
                        &source.metadata.topology_domains
                    };
                    topology.get(domain)
                }
            };
            let preferred =
                target_config.kv_transfer_enforcement == Some(KvTransferEnforcement::Preferred);
            candidates.best_ranked_source_prefix(target_cached_prefix_blocks, |source| {
                if !is_eligible(source) {
                    return None;
                }
                let same_island = target_island.is_some() && source_island(source) == target_island;
                (same_island || preferred).then_some(same_island)
            })
        }
    }
}

/// The fetch payload for `source`'s first `prefix_blocks` blocks. The endpoint comes from the
/// snapshot that chose the source; an empty endpoint or prefix gets no payload.
fn fetch_payload(
    configs: &HashMap<WorkerId, SelectionWorkerConfig>,
    source: KvTransferCandidateSource,
    prefix_blocks: usize,
    candidates: &KvTransferCandidates,
) -> Option<KvSourceLocationsPayload> {
    let source_control_endpoint = match source {
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
    let block_hashes = candidates.block_hashes.get(..prefix_blocks)?;
    if source_control_endpoint.is_empty() || block_hashes.is_empty() {
        return None;
    }
    Some(KvSourceLocationsPayload {
        source_control_endpoint,
        block_hashes: block_hashes.to_vec(),
    })
}

/// The fetch payload Dynamo attached before hint policies: the best source for `target` beyond
/// `target_cached_prefix_blocks`. Kept so the source-rule tests exercise the same rules.
#[cfg(test)]
pub(super) fn transfer_hint_for_selection(
    configs: &HashMap<WorkerId, SelectionWorkerConfig>,
    target: WorkerWithDpRank,
    target_cached_prefix_blocks: u32,
    candidates: Option<&KvTransferCandidates>,
) -> Option<KvSourceLocationsPayload> {
    let candidates = candidates?;
    let prefix_blocks = usize::try_from(target_cached_prefix_blocks).unwrap_or(usize::MAX);
    let (source, prefix_blocks) = best_fetch_source(configs, target, prefix_blocks, candidates)?;
    fetch_payload(configs, source, prefix_blocks, candidates)
}

/// One configured hint policy of a partition, in configured order.
pub(super) struct HintPolicySlot {
    pub(super) name: Arc<str>,
    /// `None` for a policy without a KV-action stage.
    pub(super) actions: Option<Arc<dyn ActionStage>>,
}

/// Formulate one booking's KV hint: the fetch committed during selection, if any, then its
/// partition's hint policies' actions in configured order.
///
/// The router materializes the committed fetch itself, from the source and prefix that earned its
/// credit, so no policy can drop or re-decide it. A policy's output is accepted or rejected whole.
/// Output is rejected when `formulate` fails, an action ID repeats, an action other than the
/// router's `kv.fetch` uses its ID, a second `kv.fetch` appears, or the hint would exceed
/// [`MAX_KV_HINT_ACTIONS`]. The output of policies that committed a plan or fetch for the selected
/// worker is checked first, so no other policy's output can crowd it out; rejecting it fails the
/// booking, because the plan or fetch may have decided worker selection. Rejecting any other
/// output drops only that policy's actions.
pub(super) fn formulate_kv_hint(
    policies: &[HintPolicySlot],
    mut plans: HintPlans,
    inputs: &ActionInputs<'_>,
) -> Result<Option<KvHint>, SelectionError> {
    if plans.worker().is_some_and(|worker| worker != inputs.worker) {
        return Err(SelectionError::Internal(format!(
            "hint plans were made for worker {:?}, but the request booked {:?}",
            plans.worker(),
            inputs.worker
        )));
    }
    let router_fetch = committed_fetch(policies, &plans, inputs)?;
    let mut outputs = Vec::with_capacity(policies.len());
    for (index, policy) in policies.iter().enumerate() {
        let plan = plans.take_plan(index);
        let committed = plan.is_some() || plans.fetch_for(index).is_some();
        let Some(stage) = &policy.actions else {
            continue;
        };
        let context = inputs.context(stage.requirements(), router_fetch.is_some());
        match stage.formulate(&context, plan) {
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

/// The `kv.fetch` action for the fetch a policy committed during selection, if any.
fn committed_fetch(
    policies: &[HintPolicySlot],
    plans: &HintPlans,
    inputs: &ActionInputs<'_>,
) -> Result<Option<KvHintAction>, SelectionError> {
    let Some((policy, source)) = policies
        .iter()
        .enumerate()
        .find_map(|(index, policy)| Some((policy, plans.fetch_for(index)?)))
    else {
        return Ok(None);
    };
    // Selection and formulation read the same source rules and block chain, so a committed
    // source always has a payload; a missing one is a host bug, not a policy choice.
    inputs
        .fetch
        .as_ref()
        .and_then(|view| view.action(&source))
        .map(Some)
        .ok_or_else(|| {
            SelectionError::Internal(format!(
                "hint policy {:?} committed a fetch the router cannot materialize",
                policy.name
            ))
        })
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
