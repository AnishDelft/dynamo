// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hint-policy selection stages inside the composed worker-selection pipeline.

use std::collections::HashMap;
use std::sync::Arc;

use rustc_hash::FxHashMap;

use super::test_support::*;
use super::*;
use crate::plugins::hint_policy::runtime::HintSelection;
use crate::plugins::hint_policy::{
    HintPolicy, HintPolicyError, HintPolicyInstance, HintRequirements, HintSelectionContext,
    SelectionAdvice,
};
use crate::plugins::worker_selection::{RuntimePolicyUpdate, WorkerInputView};
use crate::scheduling::ScheduleMode;
use crate::scheduling::config::KvRouterConfig;

/// Costs each worker its ID, so worker 0 wins unless a hint policy intervenes.
struct WorkerIdScorer;

impl WorkerScorer for WorkerIdScorer {
    fn score(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError> {
        for (candidate, cost) in candidates.iter().zip(costs) {
            *cost = candidate.worker().worker_id as f64;
        }
        Ok(())
    }
}

struct MinCostPicker;

impl WorkerPicker for MinCostPicker {
    fn pick(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        Ok(input
            .candidates()
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| left.cost().total_cmp(&right.cost()))
            .map(|(row, _)| row)
            .unwrap_or(0))
    }
}

/// Excludes, discounts, and plans by worker ID. Its plan records the worker it was made for.
#[derive(Default)]
struct Steer {
    exclude: Vec<u64>,
    discount: Option<(u64, f64)>,
    invalid_row: bool,
    nan_cost: bool,
}

impl HintPolicy for Steer {
    type Plan = u64;

    fn requirements(&self) -> HintRequirements {
        HintRequirements::new()
            .with_selection(WorkerInputs::NONE)
            .with_actions()
    }

    fn advise_selection(
        &self,
        _context: &HintSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        advice: &mut SelectionAdvice<'_, u64>,
    ) -> Result<(), HintPolicyError> {
        if self.invalid_row {
            advice.exclude(candidates.len())?;
        }
        for (row, candidate) in candidates.iter().enumerate() {
            let worker_id = candidate.worker().worker_id;
            if self.exclude.contains(&worker_id) {
                advice.exclude(row)?;
            }
            if let Some((target, discount)) = self.discount
                && target == worker_id
            {
                advice.add_cost(row, -discount)?;
            }
            if self.nan_cost {
                advice.add_cost(row, f64::NAN)?;
            }
            advice.plan(row, worker_id)?;
        }
        Ok(())
    }
}

fn hints(policies: Vec<Steer>) -> HintSelection {
    HintSelection::new(
        policies
            .into_iter()
            .enumerate()
            .map(|(index, policy)| {
                let stages = HintPolicyInstance::new("steer", policy).into_stages();
                (index, stages.selection.expect("selection stage"))
            })
            .collect(),
    )
}

fn policy() -> WorkerSelectionPolicy {
    WorkerSelectionPolicy::new(
        KvRouterConfig::default(),
        "test",
        vec![Box::new(WorkerIdScorer)],
        Box::new(MinCostPicker),
    )
}

fn workers() -> HashMap<u64, TaintedWorkerConfig> {
    (0..3)
        .map(|worker_id| (worker_id, TaintedWorkerConfig::default()))
        .collect()
}

fn select(
    policy: &WorkerSelectionPolicy,
    request: &SchedulingRequest,
) -> Result<WorkerSelectionResult, KvSchedulerError> {
    let workers = workers();
    policy.select_worker(WorkerSelectionInput::configured(
        &workers,
        request,
        request.eligibility(),
        16,
    ))
}

fn winning_plan(result: &mut WorkerSelectionResult, policy: usize) -> Option<u64> {
    result
        .hint_plans
        .take_plan(policy)
        .map(|plan| *plan.downcast::<u64>().expect("Steer plans are worker IDs"))
}

#[test]
fn exclusions_and_costs_apply_before_the_picker_and_only_the_winner_keeps_its_plan() {
    let request = base_request(64);
    let result = select(&policy(), &request).unwrap();
    assert_eq!(result.worker.worker_id, 0);
    assert!(result.hint_plans.is_empty(), "no hint policy installed");

    // An exclusion behaves like a filter.
    let mut result = select(
        &policy().with_hint_selection(hints(vec![Steer {
            exclude: vec![0],
            ..Steer::default()
        }])),
        &request,
    )
    .unwrap();
    assert_eq!(result.worker.worker_id, 1);
    assert_eq!(winning_plan(&mut result, 0), Some(1));

    // A cost contribution is added to the configured scorers' total before picking.
    let mut result = select(
        &policy().with_hint_selection(hints(vec![Steer {
            discount: Some((2, 10.0)),
            ..Steer::default()
        }])),
        &request,
    )
    .unwrap();
    assert_eq!(result.worker.worker_id, 2);
    assert_eq!(winning_plan(&mut result, 0), Some(2));
    // Losing candidates' plans never leave the selector.
    assert_eq!(winning_plan(&mut result, 0), None);
}

#[test]
fn policies_compose_in_order_without_restoring_excluded_workers() {
    let request = base_request(64);
    let mut result = select(
        &policy().with_hint_selection(hints(vec![
            Steer {
                exclude: vec![0],
                ..Steer::default()
            },
            // Discounting an excluded worker cannot bring it back.
            Steer {
                discount: Some((0, 100.0)),
                ..Steer::default()
            },
        ])),
        &request,
    )
    .unwrap();
    assert_eq!(result.worker.worker_id, 1);
    // Each policy's plan for the winner is carried separately.
    assert_eq!(winning_plan(&mut result, 0), Some(1));
    assert_eq!(winning_plan(&mut result, 1), Some(1));
}

#[test]
fn excluding_every_candidate_reports_them_filtered() {
    let request = base_request(64);
    let error = select(
        &policy().with_hint_selection(hints(vec![Steer {
            exclude: vec![0, 1, 2],
            ..Steer::default()
        }])),
        &request,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        KvSchedulerError::AllEligibleWorkersFiltered
    ));
}

#[test]
fn invalid_advice_fails_the_selection() {
    let request = base_request(64);
    for steer in [
        Steer {
            invalid_row: true,
            ..Steer::default()
        },
        Steer {
            nan_cost: true,
            ..Steer::default()
        },
    ] {
        let error = select(&policy().with_hint_selection(hints(vec![steer])), &request)
            .unwrap_err()
            .to_string();
        assert!(error.contains("hint policy \"steer\""), "{error}");
    }
}

#[test]
fn excluded_workers_leave_the_best_eligible_cache_reuse() {
    let mut request = base_request(64);
    request.mode = ScheduleMode::Tracked {
        request_id: "tracked".into(),
    };
    request.overlap.tier_overlap_blocks.device = FxHashMap::from_iter([
        (WorkerWithDpRank::from_worker_id(0), 4),
        (WorkerWithDpRank::from_worker_id(1), 1),
    ]);
    let result = select(
        &policy().with_hint_selection(hints(vec![Steer {
            exclude: vec![0],
            ..Steer::default()
        }])),
        &request,
    )
    .unwrap();
    assert_eq!(result.max_raw_cached_tokens, Some(16));
}

#[test]
fn runtime_policy_updates_keep_hint_stages() {
    let builder: crate::plugins::worker_selection::RuntimePolicyBuilder =
        Arc::new(|_| Ok(policy()));
    let mut selector = policy()
        .with_runtime_config(
            serde_json::Map::from_iter([("weight".to_string(), serde_json::json!(1))]),
            builder,
        )
        .with_hint_selection(hints(vec![Steer {
            exclude: vec![0],
            ..Steer::default()
        }]));
    selector
        .update_runtime_config(RuntimePolicyUpdate::patch(
            0,
            serde_json::Map::from_iter([("weight".to_string(), serde_json::json!(2))]),
        ))
        .unwrap();
    let request = base_request(64);
    assert_eq!(select(&selector, &request).unwrap().worker.worker_id, 1);
}

#[test]
fn hint_stages_read_fetchable_prefixes_and_exclusions_keep_scorer_rows_aligned() {
    use std::sync::Mutex;

    use crate::kv_hints::KvTransferCandidates;
    use crate::protocols::{ExternalSequenceBlockHash, KvHintTransferWorkerMetadata};

    /// A worker that consumes router hints.
    struct HintWorker;

    impl WorkerConfigLike for HintWorker {
        fn data_parallel_start_rank(&self) -> u32 {
            0
        }

        fn data_parallel_size(&self) -> u32 {
            1
        }

        fn max_num_batched_tokens(&self) -> Option<u64> {
            None
        }

        fn total_kv_blocks(&self) -> Option<u64> {
            None
        }

        fn kv_hint_transfer_metadata_for_dp_rank(
            &self,
            _dp_rank: u32,
        ) -> Option<KvHintTransferWorkerMetadata<'_>> {
            Some(KvHintTransferWorkerMetadata {
                worker_type: "prefill",
                source_control_endpoint: Some("tcp://worker:23280"),
            })
        }
    }

    type Seen = Arc<Mutex<Vec<(u64, Option<u32>)>>>;

    fn record(seen: &Seen, candidate: WorkerCandidate<'_>) -> Option<u32> {
        let fetchable = candidate
            .kv_transfer()
            .and_then(|transfer| transfer.source_prefix_blocks());
        seen.lock()
            .unwrap()
            .push((candidate.worker().worker_id, fetchable));
        fetchable
    }

    /// Excludes every candidate that would need a KV fetch.
    struct AvoidFetches(Seen);

    impl HintPolicy for AvoidFetches {
        type Plan = ();

        fn requirements(&self) -> HintRequirements {
            HintRequirements::new().with_selection(WorkerInputs::KV_TRANSFER)
        }

        fn advise_selection(
            &self,
            _context: &HintSelectionContext<'_>,
            candidates: WorkerCandidates<'_>,
            advice: &mut SelectionAdvice<'_, ()>,
        ) -> Result<(), HintPolicyError> {
            for (row, candidate) in candidates.iter().enumerate() {
                if record(&self.0, candidate).is_some() {
                    advice.exclude(row)?;
                }
            }
            Ok(())
        }
    }

    /// Records the fetchable prefix of every row it scores.
    struct TransferScorer(Seen);

    impl WorkerScorer for TransferScorer {
        fn required_worker_inputs(&self) -> WorkerInputs {
            WorkerInputs::KV_TRANSFER
        }

        fn score(
            &mut self,
            _context: &WorkerSelectionContext<'_>,
            candidates: WorkerCandidates<'_>,
            costs: &mut [f64],
        ) -> Result<(), WorkerSelectionPolicyError> {
            for (candidate, cost) in candidates.iter().zip(costs) {
                record(&self.0, candidate);
                *cost = 0.0;
            }
            Ok(())
        }
    }

    // Worker 0 holds a three-block prefix that worker 1 could fetch.
    let warm = WorkerWithDpRank::from_worker_id(0);
    let workers = HashMap::from([(0, HintWorker), (1, HintWorker)]);
    let mut request = base_request(64);
    request.overlap.tier_overlap_blocks.device = FxHashMap::from_iter([(warm, 3)]);
    request.kv_transfer_candidates = Some(KvTransferCandidates {
        block_hashes: (101..=104).map(ExternalSequenceBlockHash).collect(),
        owner_prefix_blocks: vec![(warm.into(), 3)],
        routing_snapshot: None,
    });
    let (advised, scored) = (Seen::default(), Seen::default());
    let stages =
        HintPolicyInstance::new("avoid_fetches", AvoidFetches(Arc::clone(&advised))).into_stages();
    let policy = WorkerSelectionPolicy::new(
        KvRouterConfig::default(),
        "test",
        vec![Box::new(TransferScorer(Arc::clone(&scored)))],
        Box::new(MinCostPicker),
    )
    .with_hint_selection(HintSelection::new(vec![(
        0,
        stages.selection.expect("selection stage"),
    )]));
    let result = policy
        .select_worker(WorkerSelectionInput::configured(
            &workers,
            &request,
            request.eligibility(),
            16,
        ))
        .unwrap();
    assert_eq!(result.worker.worker_id, 0);

    let sorted = |seen: Seen| {
        let mut seen = seen.lock().unwrap().clone();
        seen.sort_unstable();
        seen
    };
    // The hint stage sees the fetchable prefixes scorers see: worker 1 could fetch three blocks.
    assert_eq!(sorted(advised), vec![(0, None), (1, Some(3))]);
    // Excluding worker 1 drops its fetch input with its row.
    assert_eq!(sorted(scored), vec![(0, None)]);
}
