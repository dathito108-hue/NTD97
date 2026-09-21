#![forbid(unsafe_code)]

use crate::{
    CognitiveContext, CognitiveCycleReport, CognitiveDirective, CognitiveError, CognitiveExecutor,
    CognitiveObservation, CognitivePlanner, CognitiveRuntime, CognitiveSignals, CognitiveVerifier,
    VerificationDecision,
};

pub trait NativeReasoningProbe {
    fn probe(&mut self, context: CognitiveContext<'_>) -> Result<CognitiveObservation, String>;
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BudgetedReasoningPlanner;

impl CognitivePlanner for BudgetedReasoningPlanner {
    fn plan(&mut self, context: CognitiveContext<'_>) -> Result<CognitiveDirective, String> {
        Ok(CognitiveDirective::Work {
            instruction: format!(
                "native reasoning probe iteration {} budget {:?}",
                context.iteration, context.budget
            ),
        })
    }
}

pub struct NativeReasoningExecutor<'a, P>
where
    P: NativeReasoningProbe,
{
    probe: &'a mut P,
}

impl<'a, P> NativeReasoningExecutor<'a, P>
where
    P: NativeReasoningProbe,
{
    pub fn new(probe: &'a mut P) -> Self {
        Self { probe }
    }
}

impl<P> CognitiveExecutor for NativeReasoningExecutor<'_, P>
where
    P: NativeReasoningProbe,
{
    fn execute(
        &mut self,
        directive: &CognitiveDirective,
        context: CognitiveContext<'_>,
    ) -> Result<CognitiveObservation, String> {
        match directive {
            CognitiveDirective::Work { .. } => self.probe.probe(context),
            _ => Err("budgeted native reasoning executor only accepts Work directives".into()),
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct NativeReasoningVerifier;

impl CognitiveVerifier for NativeReasoningVerifier {
    fn verify(
        &mut self,
        directive: &CognitiveDirective,
        observation: &CognitiveObservation,
        _context: CognitiveContext<'_>,
    ) -> VerificationDecision {
        if !matches!(directive, CognitiveDirective::Work { .. }) {
            return VerificationDecision::Reject {
                reason: "unexpected directive in native reasoning loop".into(),
            };
        }
        if observation.summary.trim().is_empty() || observation.evidence.is_empty() {
            return VerificationDecision::Reject {
                reason: "native reasoning probe returned unverified evidence".into(),
            };
        }
        VerificationDecision::Accept
    }
}

pub fn run_budgeted_reasoning_cycle<P>(
    runtime: &mut CognitiveRuntime,
    task_id: u64,
    signals: CognitiveSignals,
    probe: &mut P,
) -> Result<CognitiveCycleReport, CognitiveError>
where
    P: NativeReasoningProbe,
{
    let mut planner = BudgetedReasoningPlanner;
    let mut executor = NativeReasoningExecutor::new(probe);
    let mut verifier = NativeReasoningVerifier;
    runtime.advance(task_id, signals, &mut planner, &mut executor, &mut verifier)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CognitiveIdentity, TaskStatus};
    use ntd_core::{Intent, ReasoningBudget, TaskGraph};

    #[derive(Default)]
    struct Probe {
        calls: u32,
    }

    impl NativeReasoningProbe for Probe {
        fn probe(&mut self, context: CognitiveContext<'_>) -> Result<CognitiveObservation, String> {
            self.calls = self.calls.saturating_add(1);
            Ok(CognitiveObservation {
                summary: format!("probe {}", context.iteration),
                evidence: vec![format!("native-probe:{}", context.iteration)],
            })
        }
    }

    fn run(signals: CognitiveSignals) -> (CognitiveCycleReport, u32, TaskStatus) {
        let mut runtime = CognitiveRuntime::new(CognitiveIdentity(*b"NTD97-REASONING1"));
        let task = runtime
            .state_mut()
            .submit_task(Intent::new("reason"), TaskGraph::default(), None)
            .expect("task");
        let mut probe = Probe::default();
        let report =
            run_budgeted_reasoning_cycle(&mut runtime, task, signals, &mut probe).expect("cycle");
        let status = runtime.state().tasks.get(&task).expect("task").status;
        (report, probe.calls, status)
    }

    #[test]
    fn reflex_budget_runs_exactly_one_verified_probe() {
        let (report, calls, status) = run(CognitiveSignals {
            complexity: 0.1,
            uncertainty: 0.1,
            prior_failure: false,
        });
        assert_eq!(report.budget, ReasoningBudget::Reflex);
        assert_eq!(report.iterations, 1);
        assert_eq!(calls, 1);
        assert_eq!(status, TaskStatus::Running);
    }

    #[test]
    fn standard_budget_runs_two_verified_probes() {
        let (report, calls, _) = run(CognitiveSignals {
            complexity: 0.4,
            uncertainty: 0.1,
            prior_failure: false,
        });
        assert_eq!(report.budget, ReasoningBudget::Standard);
        assert_eq!(report.iterations, 2);
        assert_eq!(calls, 2);
    }

    #[test]
    fn deep_budget_runs_four_verified_probes() {
        let (report, calls, _) = run(CognitiveSignals {
            complexity: 0.8,
            uncertainty: 0.1,
            prior_failure: false,
        });
        assert_eq!(report.budget, ReasoningBudget::Deep);
        assert_eq!(report.iterations, 4);
        assert_eq!(calls, 4);
    }

    #[test]
    fn recovery_budget_runs_three_verified_probes() {
        let (report, calls, _) = run(CognitiveSignals {
            complexity: 0.0,
            uncertainty: 0.0,
            prior_failure: true,
        });
        assert_eq!(report.budget, ReasoningBudget::Recovery);
        assert_eq!(report.iterations, 3);
        assert_eq!(calls, 3);
    }

    #[test]
    fn verifier_rejects_probe_without_evidence() {
        struct EmptyProbe;
        impl NativeReasoningProbe for EmptyProbe {
            fn probe(
                &mut self,
                _context: CognitiveContext<'_>,
            ) -> Result<CognitiveObservation, String> {
                Ok(CognitiveObservation::new("missing evidence"))
            }
        }

        let mut runtime = CognitiveRuntime::new(CognitiveIdentity(*b"NTD97-REASONING1"));
        let task = runtime
            .state_mut()
            .submit_task(Intent::new("reason"), TaskGraph::default(), None)
            .expect("task");
        let report = run_budgeted_reasoning_cycle(
            &mut runtime,
            task,
            CognitiveSignals {
                complexity: 0.1,
                uncertainty: 0.1,
                prior_failure: false,
            },
            &mut EmptyProbe,
        )
        .expect("cycle");

        assert_eq!(report.status, TaskStatus::Failed);
        assert_eq!(report.iterations, 1);
    }
}
