#![forbid(unsafe_code)]

/// Compute depth selected by the cognitive loop.
///
/// Response length is intentionally independent from this value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningBudget {
    Reflex,
    Standard,
    Deep,
    Recovery,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SideEffectClass {
    ReadOnly,
    Reversible,
    ExternalWrite,
    Irreversible,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Intent {
    pub objective: String,
    pub completion_criteria: Vec<String>,
    pub max_steps: u32,
}

impl Intent {
    pub fn new(objective: impl Into<String>) -> Self {
        Self {
            objective: objective.into(),
            completion_criteria: Vec::new(),
            max_steps: 32,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CapabilityId(pub String);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionNode {
    pub id: u32,
    pub capability: CapabilityId,
    pub side_effect: SideEffectClass,
    pub verification_required: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TaskGraph {
    pub actions: Vec<ActionNode>,
}

impl TaskGraph {
    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intent_defaults_to_bounded_execution() {
        let intent = Intent::new("create a verified artifact");
        assert_eq!(intent.max_steps, 32);
    }
}
