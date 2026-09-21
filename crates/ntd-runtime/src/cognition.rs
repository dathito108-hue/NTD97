#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use ntd_core::{Intent, ReasoningBudget, TaskGraph};

use crate::{
    choose_reasoning_budget, CognitiveSignals, MemoryError, MemoryKind, MemoryQuery,
    SovereignMemory,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CognitiveIdentity(pub [u8; 16]);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum GoalStatus {
    Active = 1,
    Completed = 2,
    Paused = 3,
    Failed = 4,
}

impl TryFrom<u8> for GoalStatus {
    type Error = CognitiveError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Active),
            2 => Ok(Self::Completed),
            3 => Ok(Self::Paused),
            4 => Ok(Self::Failed),
            other => Err(CognitiveError::InvalidGoalStatus(other)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Goal {
    pub id: u64,
    pub objective: String,
    pub completion_criteria: Vec<String>,
    pub status: GoalStatus,
    pub created_tick: u64,
    pub updated_tick: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TaskStatus {
    Planned = 1,
    Running = 2,
    Completed = 3,
    Failed = 4,
    Paused = 5,
}

impl TryFrom<u8> for TaskStatus {
    type Error = CognitiveError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Planned),
            2 => Ok(Self::Running),
            3 => Ok(Self::Completed),
            4 => Ok(Self::Failed),
            5 => Ok(Self::Paused),
            other => Err(CognitiveError::InvalidTaskStatus(other)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CognitiveTask {
    pub id: u64,
    pub goal_id: Option<u64>,
    pub intent: Intent,
    pub graph: TaskGraph,
    pub status: TaskStatus,
    pub steps_taken: u32,
    pub verification_failures: u32,
    pub last_observation: Option<String>,
    pub created_tick: u64,
    pub updated_tick: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorldFact {
    pub key: String,
    pub value: String,
    pub revision: u64,
    pub updated_tick: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LearnedDelta {
    pub id: u64,
    pub namespace: String,
    pub generation: u64,
    pub payload: Vec<u8>,
    pub active: bool,
}

pub trait DeltaActivationHook {
    fn activate(&mut self, delta: &LearnedDelta) -> Result<(), String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CognitiveState {
    pub identity: CognitiveIdentity,
    pub tick: u64,
    pub next_goal_id: u64,
    pub next_task_id: u64,
    pub next_delta_id: u64,
    pub goals: BTreeMap<u64, Goal>,
    pub tasks: BTreeMap<u64, CognitiveTask>,
    pub world: BTreeMap<String, WorldFact>,
    pub memory: SovereignMemory,
    pub deltas: BTreeMap<u64, LearnedDelta>,
}

impl CognitiveState {
    pub fn new(identity: CognitiveIdentity) -> Self {
        Self {
            identity,
            tick: 0,
            next_goal_id: 1,
            next_task_id: 1,
            next_delta_id: 1,
            goals: BTreeMap::new(),
            tasks: BTreeMap::new(),
            world: BTreeMap::new(),
            memory: SovereignMemory::new(),
            deltas: BTreeMap::new(),
        }
    }

    pub fn add_goal(
        &mut self,
        objective: impl Into<String>,
        completion_criteria: Vec<String>,
    ) -> Result<u64, CognitiveError> {
        let objective = objective.into();
        if objective.trim().is_empty() {
            return Err(CognitiveError::EmptyObjective);
        }

        let id = self.next_goal_id;
        self.next_goal_id = self
            .next_goal_id
            .checked_add(1)
            .ok_or(CognitiveError::Overflow)?;
        self.goals.insert(
            id,
            Goal {
                id,
                objective,
                completion_criteria,
                status: GoalStatus::Active,
                created_tick: self.tick,
                updated_tick: self.tick,
            },
        );
        Ok(id)
    }

    pub fn submit_task(
        &mut self,
        intent: Intent,
        graph: TaskGraph,
        goal_id: Option<u64>,
    ) -> Result<u64, CognitiveError> {
        if intent.objective.trim().is_empty() {
            return Err(CognitiveError::EmptyObjective);
        }
        if intent.max_steps == 0 {
            return Err(CognitiveError::InvalidStepBudget);
        }
        match goal_id {
            Some(goal_id) if !self.goals.contains_key(&goal_id) => {
                return Err(CognitiveError::MissingGoal(goal_id));
            }
            _ => {}
        }

        let id = self.next_task_id;
        self.next_task_id = self
            .next_task_id
            .checked_add(1)
            .ok_or(CognitiveError::Overflow)?;
        self.tasks.insert(
            id,
            CognitiveTask {
                id,
                goal_id,
                intent,
                graph,
                status: TaskStatus::Planned,
                steps_taken: 0,
                verification_failures: 0,
                last_observation: None,
                created_tick: self.tick,
                updated_tick: self.tick,
            },
        );
        Ok(id)
    }

    pub fn set_world_fact(
        &mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<(), CognitiveError> {
        let key = key.into();
        if key.trim().is_empty() {
            return Err(CognitiveError::EmptyWorldKey);
        }
        let value = value.into();

        let revision = self
            .world
            .get(&key)
            .map(|fact| fact.revision.saturating_add(1))
            .unwrap_or(1);
        self.world.insert(
            key.clone(),
            WorldFact {
                key,
                value,
                revision,
                updated_tick: self.tick,
            },
        );
        Ok(())
    }

    pub fn install_delta(
        &mut self,
        namespace: impl Into<String>,
        generation: u64,
        payload: Vec<u8>,
    ) -> Result<u64, CognitiveError> {
        let namespace = namespace.into();
        if namespace.trim().is_empty() {
            return Err(CognitiveError::EmptyDeltaNamespace);
        }

        let id = self.next_delta_id;
        self.next_delta_id = self
            .next_delta_id
            .checked_add(1)
            .ok_or(CognitiveError::Overflow)?;
        self.deltas.insert(
            id,
            LearnedDelta {
                id,
                namespace,
                generation,
                payload,
                active: false,
            },
        );
        Ok(id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CognitiveDirective {
    Recall {
        query: MemoryQuery,
    },
    RecordMemory {
        kind: MemoryKind,
        content: String,
        tags: Vec<String>,
        importance: u16,
    },
    UpdateWorld {
        key: String,
        value: String,
    },
    Work {
        instruction: String,
    },
    Complete {
        summary: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CognitiveObservation {
    pub summary: String,
    pub evidence: Vec<String>,
}

impl CognitiveObservation {
    pub fn new(summary: impl Into<String>) -> Self {
        Self {
            summary: summary.into(),
            evidence: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationDecision {
    Accept,
    Retry { reason: String },
    Reject { reason: String },
}

pub struct CognitiveContext<'a> {
    pub state: &'a CognitiveState,
    pub task: &'a CognitiveTask,
    pub budget: ReasoningBudget,
    pub iteration: u32,
}

pub trait CognitivePlanner {
    fn plan(&mut self, context: CognitiveContext<'_>) -> Result<CognitiveDirective, String>;
}

pub trait CognitiveExecutor {
    fn execute(
        &mut self,
        directive: &CognitiveDirective,
        context: CognitiveContext<'_>,
    ) -> Result<CognitiveObservation, String>;
}

pub trait CognitiveVerifier {
    fn verify(
        &mut self,
        directive: &CognitiveDirective,
        observation: &CognitiveObservation,
        context: CognitiveContext<'_>,
    ) -> VerificationDecision;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CognitiveCycleReport {
    pub task_id: u64,
    pub budget: ReasoningBudget,
    pub iterations: u32,
    pub status: TaskStatus,
    pub verification_failures: u32,
    pub last_observation: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CognitiveError {
    EmptyObjective,
    EmptyWorldKey,
    EmptyDeltaNamespace,
    InvalidStepBudget,
    MissingGoal(u64),
    MissingTask(u64),
    MissingDelta(u64),
    InvalidGoalStatus(u8),
    InvalidTaskStatus(u8),
    Planner(String),
    Executor(String),
    Memory(MemoryError),
    Overflow,
}

impl From<MemoryError> for CognitiveError {
    fn from(value: MemoryError) -> Self {
        Self::Memory(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CognitiveRuntime {
    state: CognitiveState,
}

impl CognitiveRuntime {
    pub fn new(identity: CognitiveIdentity) -> Self {
        Self {
            state: CognitiveState::new(identity),
        }
    }

    pub fn from_state(state: CognitiveState) -> Result<Self, CognitiveError> {
        validate_state(&state)?;
        Ok(Self { state })
    }

    pub fn state(&self) -> &CognitiveState {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut CognitiveState {
        &mut self.state
    }

    pub fn start_external_task(&mut self, task_id: u64) -> Result<(), CognitiveError> {
        self.state.tick = self
            .state
            .tick
            .checked_add(1)
            .ok_or(CognitiveError::Overflow)?;
        let task = self
            .state
            .tasks
            .get_mut(&task_id)
            .ok_or(CognitiveError::MissingTask(task_id))?;
        if matches!(task.status, TaskStatus::Completed | TaskStatus::Failed) {
            return Ok(());
        }
        task.status = TaskStatus::Running;
        task.updated_tick = self.state.tick;
        Ok(())
    }

    pub fn complete_external_task(
        &mut self,
        task_id: u64,
        summary: impl Into<String>,
    ) -> Result<(), CognitiveError> {
        let summary = summary.into();
        if summary.trim().is_empty() {
            return Err(CognitiveError::Executor(
                "external task completion summary is empty".into(),
            ));
        }

        self.state.tick = self
            .state
            .tick
            .checked_add(1)
            .ok_or(CognitiveError::Overflow)?;
        let observation = CognitiveObservation::new(summary.clone());
        self.commit_directive(
            task_id,
            &CognitiveDirective::Complete { summary },
            &observation,
        )
    }

    pub fn pause_external_task(
        &mut self,
        task_id: u64,
        reason: impl Into<String>,
    ) -> Result<(), CognitiveError> {
        let reason = reason.into();
        if reason.trim().is_empty() {
            return Err(CognitiveError::Executor(
                "external task pause reason is empty".into(),
            ));
        }

        self.state.tick = self
            .state
            .tick
            .checked_add(1)
            .ok_or(CognitiveError::Overflow)?;
        let task = self
            .state
            .tasks
            .get_mut(&task_id)
            .ok_or(CognitiveError::MissingTask(task_id))?;
        if matches!(task.status, TaskStatus::Completed | TaskStatus::Failed) {
            return Ok(());
        }
        task.status = TaskStatus::Paused;
        task.updated_tick = self.state.tick;
        task.last_observation = Some(reason.clone());
        self.state.memory.store(
            MemoryKind::Episodic,
            format!("task {task_id} paused: {reason}"),
            vec!["task".into(), "pause".into()],
            400,
            self.state.tick,
        )?;
        Ok(())
    }

    pub fn activate_delta<H: DeltaActivationHook>(
        &mut self,
        delta_id: u64,
        hook: &mut H,
    ) -> Result<(), CognitiveError> {
        let delta = self
            .state
            .deltas
            .get(&delta_id)
            .cloned()
            .ok_or(CognitiveError::MissingDelta(delta_id))?;
        hook.activate(&delta).map_err(CognitiveError::Executor)?;
        let stored = self
            .state
            .deltas
            .get_mut(&delta_id)
            .ok_or(CognitiveError::MissingDelta(delta_id))?;
        stored.active = true;
        Ok(())
    }

    pub fn advance<P, E, V>(
        &mut self,
        task_id: u64,
        signals: CognitiveSignals,
        planner: &mut P,
        executor: &mut E,
        verifier: &mut V,
    ) -> Result<CognitiveCycleReport, CognitiveError>
    where
        P: CognitivePlanner,
        E: CognitiveExecutor,
        V: CognitiveVerifier,
    {
        let budget = choose_reasoning_budget(signals);
        let max_iterations = iterations_for_budget(budget);

        {
            let task = self
                .state
                .tasks
                .get_mut(&task_id)
                .ok_or(CognitiveError::MissingTask(task_id))?;
            if matches!(task.status, TaskStatus::Completed | TaskStatus::Failed) {
                return Ok(report(task, budget, 0));
            }
            task.status = TaskStatus::Running;
        }

        let mut iterations = 0u32;
        for iteration in 0..max_iterations {
            let max_steps_reached = {
                let task = self
                    .state
                    .tasks
                    .get(&task_id)
                    .ok_or(CognitiveError::MissingTask(task_id))?;
                task.steps_taken >= task.intent.max_steps
            };
            if max_steps_reached {
                self.fail_task(task_id, "task step budget exhausted")?;
                break;
            }

            self.state.tick = self
                .state
                .tick
                .checked_add(1)
                .ok_or(CognitiveError::Overflow)?;

            let directive = {
                let task = self
                    .state
                    .tasks
                    .get(&task_id)
                    .ok_or(CognitiveError::MissingTask(task_id))?;
                planner
                    .plan(CognitiveContext {
                        state: &self.state,
                        task,
                        budget,
                        iteration,
                    })
                    .map_err(CognitiveError::Planner)?
            };

            let observation =
                self.execute_directive(task_id, &directive, budget, iteration, executor)?;

            let decision = {
                let task = self
                    .state
                    .tasks
                    .get(&task_id)
                    .ok_or(CognitiveError::MissingTask(task_id))?;
                verifier.verify(
                    &directive,
                    &observation,
                    CognitiveContext {
                        state: &self.state,
                        task,
                        budget,
                        iteration,
                    },
                )
            };

            iterations = iterations.saturating_add(1);
            match decision {
                VerificationDecision::Accept => {
                    self.commit_directive(task_id, &directive, &observation)?;
                    if matches!(directive, CognitiveDirective::Complete { .. }) {
                        break;
                    }
                }
                VerificationDecision::Retry { reason } => {
                    let task = self
                        .state
                        .tasks
                        .get_mut(&task_id)
                        .ok_or(CognitiveError::MissingTask(task_id))?;
                    task.verification_failures = task.verification_failures.saturating_add(1);
                    task.last_observation = Some(reason.clone());
                    task.updated_tick = self.state.tick;
                    self.state.memory.store(
                        MemoryKind::Episodic,
                        format!("verification retry: {reason}"),
                        vec!["verification".into(), "retry".into()],
                        600,
                        self.state.tick,
                    )?;
                }
                VerificationDecision::Reject { reason } => {
                    self.fail_task(task_id, &reason)?;
                    break;
                }
            }
        }

        let task = self
            .state
            .tasks
            .get(&task_id)
            .ok_or(CognitiveError::MissingTask(task_id))?;
        Ok(report(task, budget, iterations))
    }

    fn execute_directive<E: CognitiveExecutor>(
        &mut self,
        task_id: u64,
        directive: &CognitiveDirective,
        budget: ReasoningBudget,
        iteration: u32,
        executor: &mut E,
    ) -> Result<CognitiveObservation, CognitiveError> {
        match directive {
            CognitiveDirective::Recall { query } => {
                let hits = self.state.memory.retrieve(query, self.state.tick)?;
                let evidence = hits
                    .iter()
                    .map(|hit| format!("memory:{}:{}", hit.id, hit.record.content))
                    .collect::<Vec<_>>();
                Ok(CognitiveObservation {
                    summary: format!("recalled {} memories", hits.len()),
                    evidence,
                })
            }
            CognitiveDirective::RecordMemory { content, .. } => Ok(CognitiveObservation::new(
                format!("memory prepared: {content}"),
            )),
            CognitiveDirective::UpdateWorld { key, value } => Ok(CognitiveObservation::new(
                format!("world update prepared: {key}={value}"),
            )),
            CognitiveDirective::Complete { summary } => Ok(CognitiveObservation::new(format!(
                "completion prepared: {summary}"
            ))),
            CognitiveDirective::Work { .. } => {
                let task = self
                    .state
                    .tasks
                    .get(&task_id)
                    .ok_or(CognitiveError::MissingTask(task_id))?;
                executor
                    .execute(
                        directive,
                        CognitiveContext {
                            state: &self.state,
                            task,
                            budget,
                            iteration,
                        },
                    )
                    .map_err(CognitiveError::Executor)
            }
        }
    }

    fn commit_directive(
        &mut self,
        task_id: u64,
        directive: &CognitiveDirective,
        observation: &CognitiveObservation,
    ) -> Result<(), CognitiveError> {
        match directive {
            CognitiveDirective::Recall { .. } | CognitiveDirective::Work { .. } => {}
            CognitiveDirective::RecordMemory {
                kind,
                content,
                tags,
                importance,
            } => {
                self.state.memory.store(
                    *kind,
                    content.clone(),
                    tags.clone(),
                    *importance,
                    self.state.tick,
                )?;
            }
            CognitiveDirective::UpdateWorld { key, value } => {
                self.state.set_world_fact(key.clone(), value.clone())?;
            }
            CognitiveDirective::Complete { summary } => {
                let goal_id = self
                    .state
                    .tasks
                    .get(&task_id)
                    .ok_or(CognitiveError::MissingTask(task_id))?
                    .goal_id;
                if let Some(goal) = goal_id.and_then(|id| self.state.goals.get_mut(&id)) {
                    goal.status = GoalStatus::Completed;
                    goal.updated_tick = self.state.tick;
                }
                self.state.memory.store(
                    MemoryKind::Episodic,
                    format!("task {task_id} completed: {summary}"),
                    vec!["task".into(), "completion".into()],
                    700,
                    self.state.tick,
                )?;
            }
        }

        let task = self
            .state
            .tasks
            .get_mut(&task_id)
            .ok_or(CognitiveError::MissingTask(task_id))?;
        task.steps_taken = task.steps_taken.saturating_add(1);
        task.updated_tick = self.state.tick;
        task.last_observation = Some(observation.summary.clone());
        if matches!(directive, CognitiveDirective::Complete { .. }) {
            task.status = TaskStatus::Completed;
        }
        Ok(())
    }

    fn fail_task(&mut self, task_id: u64, reason: &str) -> Result<(), CognitiveError> {
        let goal_id = self
            .state
            .tasks
            .get(&task_id)
            .ok_or(CognitiveError::MissingTask(task_id))?
            .goal_id;

        let task = self
            .state
            .tasks
            .get_mut(&task_id)
            .ok_or(CognitiveError::MissingTask(task_id))?;
        task.status = TaskStatus::Failed;
        task.updated_tick = self.state.tick;
        task.last_observation = Some(reason.to_owned());

        if let Some(goal) = goal_id.and_then(|id| self.state.goals.get_mut(&id)) {
            goal.status = GoalStatus::Failed;
            goal.updated_tick = self.state.tick;
        }

        self.state.memory.store(
            MemoryKind::Episodic,
            format!("task {task_id} failed: {reason}"),
            vec!["task".into(), "failure".into()],
            800,
            self.state.tick,
        )?;
        Ok(())
    }
}

fn iterations_for_budget(budget: ReasoningBudget) -> u32 {
    match budget {
        ReasoningBudget::Reflex => 1,
        ReasoningBudget::Standard => 2,
        ReasoningBudget::Deep => 4,
        ReasoningBudget::Recovery => 3,
    }
}

fn report(task: &CognitiveTask, budget: ReasoningBudget, iterations: u32) -> CognitiveCycleReport {
    CognitiveCycleReport {
        task_id: task.id,
        budget,
        iterations,
        status: task.status,
        verification_failures: task.verification_failures,
        last_observation: task.last_observation.clone(),
    }
}

fn validate_state(state: &CognitiveState) -> Result<(), CognitiveError> {
    if state.next_goal_id == 0 || state.next_task_id == 0 || state.next_delta_id == 0 {
        return Err(CognitiveError::Overflow);
    }

    if state
        .goals
        .keys()
        .next_back()
        .is_some_and(|id| state.next_goal_id <= *id)
        || state
            .tasks
            .keys()
            .next_back()
            .is_some_and(|id| state.next_task_id <= *id)
        || state
            .deltas
            .keys()
            .next_back()
            .is_some_and(|id| state.next_delta_id <= *id)
    {
        return Err(CognitiveError::Overflow);
    }

    for (id, goal) in &state.goals {
        if *id != goal.id || goal.objective.trim().is_empty() {
            return Err(CognitiveError::Overflow);
        }
    }

    for (id, task) in &state.tasks {
        if *id != task.id
            || task.intent.objective.trim().is_empty()
            || task.intent.max_steps == 0
            || task.steps_taken > task.intent.max_steps
        {
            return Err(CognitiveError::Overflow);
        }
        match task.goal_id {
            Some(goal_id) if !state.goals.contains_key(&goal_id) => {
                return Err(CognitiveError::MissingGoal(goal_id));
            }
            _ => {}
        }
    }

    for (key, fact) in &state.world {
        if key != &fact.key || key.trim().is_empty() || fact.revision == 0 {
            return Err(CognitiveError::Overflow);
        }
    }

    for (id, delta) in &state.deltas {
        if *id != delta.id || delta.namespace.trim().is_empty() {
            return Err(CognitiveError::Overflow);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn external_completion_and_pause_use_canonical_task_transitions() {
        let identity = CognitiveIdentity(*b"NTD97-COGNITION1");
        let mut runtime = CognitiveRuntime::new(identity);
        let completed = runtime
            .state_mut()
            .submit_task(
                ntd_core::Intent::new("answer user"),
                ntd_core::TaskGraph::default(),
                None,
            )
            .expect("completed task");
        runtime.start_external_task(completed).expect("start");
        assert_eq!(
            runtime.state().tasks.get(&completed).expect("task").status,
            TaskStatus::Running
        );
        runtime
            .complete_external_task(completed, "answer committed")
            .expect("complete");
        assert_eq!(
            runtime.state().tasks.get(&completed).expect("task").status,
            TaskStatus::Completed
        );

        let paused = runtime
            .state_mut()
            .submit_task(
                ntd_core::Intent::new("interrupted user turn"),
                ntd_core::TaskGraph::default(),
                None,
            )
            .expect("paused task");
        runtime
            .pause_external_task(paused, "generation cancelled")
            .expect("pause");
        assert_eq!(
            runtime.state().tasks.get(&paused).expect("task").status,
            TaskStatus::Paused
        );
        assert!(runtime
            .state()
            .memory
            .records()
            .values()
            .any(|record| record.content.contains("generation cancelled")));
    }


    use super::*;
    use ntd_core::TaskGraph;

    struct Planner {
        call: u32,
    }

    impl CognitivePlanner for Planner {
        fn plan(&mut self, _context: CognitiveContext<'_>) -> Result<CognitiveDirective, String> {
            self.call += 1;
            Ok(match self.call {
                1 => CognitiveDirective::Work {
                    instruction: "analyze".into(),
                },
                _ => CognitiveDirective::Complete {
                    summary: "done".into(),
                },
            })
        }
    }

    struct Executor;

    impl CognitiveExecutor for Executor {
        fn execute(
            &mut self,
            _directive: &CognitiveDirective,
            _context: CognitiveContext<'_>,
        ) -> Result<CognitiveObservation, String> {
            Ok(CognitiveObservation::new("analysis complete"))
        }
    }

    struct Verifier;

    impl CognitiveVerifier for Verifier {
        fn verify(
            &mut self,
            _directive: &CognitiveDirective,
            _observation: &CognitiveObservation,
            _context: CognitiveContext<'_>,
        ) -> VerificationDecision {
            VerificationDecision::Accept
        }
    }

    #[test]
    fn deep_budget_can_complete_multi_iteration_task() {
        let mut runtime = CognitiveRuntime::new(CognitiveIdentity(*b"NTD97-COGNITION1"));
        let goal = runtime
            .state_mut()
            .add_goal("finish task", vec!["done".into()])
            .expect("goal");
        let task = runtime
            .state_mut()
            .submit_task(Intent::new("finish task"), TaskGraph::default(), Some(goal))
            .expect("task");

        let report = runtime
            .advance(
                task,
                CognitiveSignals {
                    complexity: 0.9,
                    uncertainty: 0.2,
                    prior_failure: false,
                },
                &mut Planner { call: 0 },
                &mut Executor,
                &mut Verifier,
            )
            .expect("advance");

        assert_eq!(report.budget, ReasoningBudget::Deep);
        assert_eq!(report.status, TaskStatus::Completed);
        assert_eq!(report.iterations, 2);
        assert_eq!(
            runtime.state().goals.get(&goal).expect("goal").status,
            GoalStatus::Completed
        );
    }

    #[test]
    fn reflex_budget_advances_only_one_iteration() {
        let mut runtime = CognitiveRuntime::new(CognitiveIdentity(*b"NTD97-COGNITION1"));
        let task = runtime
            .state_mut()
            .submit_task(Intent::new("think"), TaskGraph::default(), None)
            .expect("task");

        let report = runtime
            .advance(
                task,
                CognitiveSignals {
                    complexity: 0.1,
                    uncertainty: 0.1,
                    prior_failure: false,
                },
                &mut Planner { call: 0 },
                &mut Executor,
                &mut Verifier,
            )
            .expect("advance");

        assert_eq!(report.budget, ReasoningBudget::Reflex);
        assert_eq!(report.iterations, 1);
        assert_eq!(report.status, TaskStatus::Running);
    }
}
