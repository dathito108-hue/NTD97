#![forbid(unsafe_code)]

use ntd_core::{Intent, TaskGraph};
use ntd_runtime::{
    decode_cognitive_checkpoint, encode_cognitive_checkpoint, CognitiveContext, CognitiveDirective,
    CognitiveExecutor, CognitiveIdentity, CognitiveObservation, CognitivePlanner, CognitiveRuntime,
    CognitiveSignals, CognitiveVerifier, DeltaActivationHook, GoalStatus, LearnedDelta, MemoryKind,
    MemoryQuery, TaskStatus, VerificationDecision,
};

struct ContinuityPlanner;

impl CognitivePlanner for ContinuityPlanner {
    fn plan(&mut self, context: CognitiveContext<'_>) -> Result<CognitiveDirective, String> {
        match context.task.steps_taken {
            0 => Ok(CognitiveDirective::Recall {
                query: MemoryQuery::new("offline sovereign"),
            }),
            1 => Ok(CognitiveDirective::Work {
                instruction: "analyze recalled evidence".into(),
            }),
            2 => Ok(CognitiveDirective::UpdateWorld {
                key: "analysis.status".into(),
                value: "verified".into(),
            }),
            _ => Ok(CognitiveDirective::Complete {
                summary: "continuity restored and task completed".into(),
            }),
        }
    }
}

struct ContinuityExecutor;

impl CognitiveExecutor for ContinuityExecutor {
    fn execute(
        &mut self,
        directive: &CognitiveDirective,
        _context: CognitiveContext<'_>,
    ) -> Result<CognitiveObservation, String> {
        match directive {
            CognitiveDirective::Work { instruction } => Ok(CognitiveObservation {
                summary: format!("executed: {instruction}"),
                evidence: vec!["native-memory".into(), "local-runtime".into()],
            }),
            _ => Err("executor received internal directive".into()),
        }
    }
}

struct DeltaHook {
    activated: bool,
}

impl DeltaActivationHook for DeltaHook {
    fn activate(&mut self, delta: &LearnedDelta) -> Result<(), String> {
        if delta.namespace != "reasoning.local" {
            return Err("wrong delta namespace".into());
        }
        self.activated = true;
        Ok(())
    }
}

struct ContinuityVerifier;

impl CognitiveVerifier for ContinuityVerifier {
    fn verify(
        &mut self,
        directive: &CognitiveDirective,
        _observation: &CognitiveObservation,
        context: CognitiveContext<'_>,
    ) -> VerificationDecision {
        if matches!(directive, CognitiveDirective::Work { .. })
            && context.task.verification_failures == 0
        {
            VerificationDecision::Retry {
                reason: "require one verified retry".into(),
            }
        } else {
            VerificationDecision::Accept
        }
    }
}

#[test]
fn checkpoint_restore_continues_same_identity_and_task() {
    let identity = CognitiveIdentity(*b"NTD97-COGNITION1");
    let mut runtime = CognitiveRuntime::new(identity);

    let goal_id = runtime
        .state_mut()
        .add_goal(
            "complete sovereign continuity test",
            vec!["task completes after restore".into()],
        )
        .expect("goal");

    runtime
        .state_mut()
        .memory
        .store(
            MemoryKind::Semantic,
            "offline sovereign execution uses local NTD97 state",
            vec!["identity".into(), "offline".into()],
            900,
            0,
        )
        .expect("seed memory");

    let delta_id = runtime
        .state_mut()
        .install_delta("reasoning.local", 1, vec![9, 7])
        .expect("delta");
    let mut delta_hook = DeltaHook { activated: false };
    runtime
        .activate_delta(delta_id, &mut delta_hook)
        .expect("activate delta");
    assert!(delta_hook.activated);

    let task_id = runtime
        .state_mut()
        .submit_task(
            Intent::new("prove cognitive continuity"),
            TaskGraph::default(),
            Some(goal_id),
        )
        .expect("task");

    let first = runtime
        .advance(
            task_id,
            CognitiveSignals {
                complexity: 0.5,
                uncertainty: 0.2,
                prior_failure: false,
            },
            &mut ContinuityPlanner,
            &mut ContinuityExecutor,
            &mut ContinuityVerifier,
        )
        .expect("first cycle");

    assert_eq!(first.status, TaskStatus::Running);
    assert_eq!(first.iterations, 2);
    assert_eq!(first.verification_failures, 1);

    let checkpoint = encode_cognitive_checkpoint(runtime.state()).expect("checkpoint");
    let restored_state = decode_cognitive_checkpoint(&checkpoint).expect("decode checkpoint");
    let mut restored = CognitiveRuntime::from_state(restored_state).expect("restore runtime");

    assert_eq!(restored.state().identity, identity);
    assert!(
        restored
            .state()
            .deltas
            .get(&delta_id)
            .expect("delta")
            .active
    );
    assert_eq!(
        restored
            .state()
            .tasks
            .get(&task_id)
            .expect("task")
            .verification_failures,
        1
    );

    let second = restored
        .advance(
            task_id,
            CognitiveSignals {
                complexity: 0.2,
                uncertainty: 0.1,
                prior_failure: true,
            },
            &mut ContinuityPlanner,
            &mut ContinuityExecutor,
            &mut ContinuityVerifier,
        )
        .expect("recovery cycle");

    assert_eq!(second.status, TaskStatus::Completed);
    assert_eq!(second.iterations, 3);
    assert_eq!(
        restored.state().goals.get(&goal_id).expect("goal").status,
        GoalStatus::Completed
    );
    assert_eq!(
        restored
            .state()
            .world
            .get("analysis.status")
            .expect("world")
            .value,
        "verified"
    );

    let mut query = MemoryQuery::new("task completed");
    query.kinds.push(MemoryKind::Episodic);
    let recall_tick = restored.state().tick.saturating_add(1);
    let hits = restored
        .state_mut()
        .memory
        .retrieve(&query, recall_tick)
        .expect("completion memory");
    assert!(!hits.is_empty());
}
