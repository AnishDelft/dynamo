// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo's `kv_fetch` preset: fetch a longer KV prefix from a peer worker or cache owner.

use std::sync::Arc;

use serde::Deserialize;

use super::registry::{HintPolicyDefinition, HintPolicyParameters, HintPolicyProviderError};
use super::{
    HintActionContext, HintPolicy, HintPolicyError, HintPolicyInstance, HintRequirements,
    HintSelectionContext, KvFetchSource, SelectionAdvice,
};
use crate::kv_hints::KvHintAction;
use crate::plugins::worker_selection::{WorkerCandidates, WorkerInputs};

/// The registered type of Dynamo's KV fetch preset.
pub const KV_FETCH_HINT_POLICY_TYPE: &str = "kv_fetch";

/// Parameters of the `kv_fetch` preset.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct KvFetchHintPolicyParameters {
    /// Cost subtracted from a candidate for each block a fetch would add beyond everything the
    /// candidate already holds, in the worker-selection policy's cost units. Dynamo's default
    /// policy charges `overlap_score_weight` per uncached prefill block, so a credit between zero
    /// and that weight values a fetched block as partly cached.
    ///
    /// Zero keeps fetch out of worker selection: the router chooses the worker first and then
    /// fetches for it, as it did before hint policies.
    pub credit_per_block: f64,
    /// Fetch only when the source adds at least this many blocks to the target's prefix.
    pub min_additional_blocks: usize,
}

impl Default for KvFetchHintPolicyParameters {
    fn default() -> Self {
        Self {
            credit_per_block: 0.0,
            min_additional_blocks: 1,
        }
    }
}

/// Dynamo's KV fetch preset.
///
/// The host chooses each target's eligible source by its router-hint rules (worker type, control
/// endpoint, attachment, and KV transfer island). This policy decides whether the target fetches
/// from it.
///
/// With a positive `credit_per_block`, the policy decides for every candidate before scoring,
/// credits the blocks the fetch adds, and commits the winner to that same fetch. Otherwise it
/// decides only for the selected worker, after booking, and never changes worker selection.
#[derive(Clone, Debug)]
pub struct KvFetchHintPolicy {
    parameters: KvFetchHintPolicyParameters,
}

impl KvFetchHintPolicy {
    pub fn new(parameters: KvFetchHintPolicyParameters) -> Result<Self, HintPolicyProviderError> {
        if !parameters.credit_per_block.is_finite() || parameters.credit_per_block < 0.0 {
            return Err(HintPolicyProviderError::new(
                "credit_per_block must be finite and non-negative",
            ));
        }
        if parameters.min_additional_blocks == 0 {
            return Err(HintPolicyProviderError::new(
                "min_additional_blocks must be at least 1",
            ));
        }
        Ok(Self { parameters })
    }

    fn credits_selection(&self) -> bool {
        self.parameters.credit_per_block > 0.0
    }

    fn accepts(&self, source: &KvFetchSource) -> bool {
        source.additional_blocks() >= self.parameters.min_additional_blocks
    }
}

impl HintPolicy for KvFetchHintPolicy {
    type Plan = ();

    fn requirements(&self) -> HintRequirements {
        let requirements = HintRequirements::new()
            .with_actions()
            .with_kv_fetch_sources();
        if self.credits_selection() {
            requirements.with_selection(WorkerInputs::CACHE)
        } else {
            requirements
        }
    }

    fn advise_selection(
        &self,
        context: &HintSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        advice: &mut SelectionAdvice<'_, ()>,
    ) -> Result<(), HintPolicyError> {
        for (row, candidate) in candidates.iter().enumerate() {
            let Some(source) = context.kv_fetch_source(candidate) else {
                continue;
            };
            if !self.accepts(&source) {
                continue;
            }
            // Credit only blocks no local tier or shared cache already credits.
            let mut credited = source.additional_blocks() as f64;
            if let Some(cache) = candidate.cache() {
                let local = (cache.device_overlap_blocks()
                    + cache.host_overlap_blocks()
                    + cache.disk_overlap_blocks())
                .max(source.local_prefix_blocks() as f64);
                credited = (source.prefix_blocks() as f64 - local).max(0.0);
                if let Some(shared) = cache.shared_hits() {
                    let from = local.round() as u32;
                    let to = u32::try_from(source.prefix_blocks()).unwrap_or(u32::MAX);
                    let shared_in_fetch = shared
                        .hits_beyond(from)
                        .saturating_sub(shared.hits_beyond(to));
                    credited = (credited - f64::from(shared_in_fetch)).max(0.0);
                }
            }
            // A finite credit per block can still overflow for a long prefix.
            let credit = (self.parameters.credit_per_block * credited).min(f64::MAX);
            advice.fetch(row, source, credit)?;
        }
        Ok(())
    }

    fn formulate(
        &self,
        context: &HintActionContext<'_>,
        _plan: Option<()>,
    ) -> Result<Vec<KvHintAction>, HintPolicyError> {
        // A crediting policy decided during selection; the host attaches the fetch it committed.
        if self.credits_selection() {
            return Ok(Vec::new());
        }
        let Some(source) = context
            .kv_fetch_source()
            .filter(|source| self.accepts(source))
        else {
            return Ok(Vec::new());
        };
        // A source that lost its endpoint since selection gets no hint, as before hint policies.
        Ok(context.kv_fetch_action(&source).into_iter().collect())
    }
}

pub(super) fn kv_fetch_provider(
    parameters: &HintPolicyParameters,
) -> Result<HintPolicyDefinition, HintPolicyProviderError> {
    KvFetchHintPolicy::definition(parameters.deserialize()?)
}

impl KvFetchHintPolicy {
    /// A definition whose every partition runs this preset with `parameters`.
    pub fn definition(
        parameters: KvFetchHintPolicyParameters,
    ) -> Result<HintPolicyDefinition, HintPolicyProviderError> {
        let policy = Self::new(parameters)?;
        let requirements = policy.requirements();
        Ok(HintPolicyDefinition::new(
            requirements,
            Arc::new(move |_config, _worker_type, _partition| {
                HintPolicyInstance::new(KV_FETCH_HINT_POLICY_TYPE, policy.clone())
            }),
        ))
    }
}
