#![forbid(unsafe_code)]

mod action_checkpoint;
mod action_fabric;
mod capability;
mod checkpoint;
mod cognition;
mod executor;
mod generation;
mod gpt2;
mod memory;
mod mobile;
mod tensor;
mod tokenizer;

pub use action_checkpoint::{
    decode_action_fabric_checkpoint, encode_action_fabric_checkpoint, ActionCheckpointError,
    TAF97_HEADER_LEN, TAF97_MAGIC, TAF97_MAJOR, TAF97_MINOR,
};
pub use action_fabric::{
    ActionFabric, ActionFabricError, ActionFabricState, ActionId, ActionPlanId, ActionPlanState,
    ActionPlanStatus, ActionStatus, ActionStepReport, ActionVerification, ActionVerifier,
    AdapterResult, CapabilityAdapter, PlannedAction,
};
pub use capability::{
    ActionOutput, ActionValue, AuthorityGrant, AuthorityScope, CapabilityDescriptor,
    CapabilityDomain, CapabilityError, CapabilityRegistry, TypedAction,
};
pub use checkpoint::{
    decode_cognitive_checkpoint, encode_cognitive_checkpoint, CheckpointError, SIK97_HEADER_LEN,
    SIK97_MAGIC, SIK97_MAJOR, SIK97_MINOR,
};
pub use cognition::{
    CognitiveContext, CognitiveCycleReport, CognitiveDirective, CognitiveError, CognitiveExecutor,
    CognitiveIdentity, CognitiveObservation, CognitivePlanner, CognitiveRuntime, CognitiveState,
    CognitiveTask, CognitiveVerifier, DeltaActivationHook, Goal, GoalStatus, LearnedDelta,
    TaskStatus, VerificationDecision, WorldFact,
};
pub use executor::{ExecutionError, GraphExecutor};
pub use generation::{
    sample_token, DistributionKind, GeneratedText, GenerationConfig, GenerationError,
    GenerationResult, GraphGenerator, KvCache, KvCacheError, KvLayerCache, PrefixCache,
    SamplingError, SamplingMode,
};
pub use gpt2::{Gpt2BpeConfig, Gpt2BpeTokenizer};
pub use memory::{MemoryError, MemoryHit, MemoryKind, MemoryQuery, MemoryRecord, SovereignMemory};
pub use mobile::{
    npu_provider, page_windows, plan_tensor_placement, verify_provider_equivalence,
    vulkan_provider, AdaptiveExecutionProvider, AutotuneTable, ByteRegion, ComputePolicy,
    CpuReferenceMobileProvider, CpuTiledProvider, DeviceCapabilities, MobileComputeError,
    MobileExecutionProvider, PageWindow, PagedByteReader, PowerClass, ProfiledProvider,
    ProviderKind, ProviderMeasurement, ProviderProfile, QuantizationProfile, ResourceSnapshot,
    SliceByteRegion, TensorPlacement, TensorPlacementPlan, ThermalState,
};
pub use tensor::{
    CpuReferenceProvider, ExecutionProvider, QuantizationParams, Tensor, TensorError,
    TensorLoadError, TensorLoader,
};
pub use tokenizer::{
    LlamaSpmConfig, LlamaSpmTokenizer, TextTokenizer, TokenizerError, VocabularyTokenizer,
    TOKEN_TYPE_BYTE, TOKEN_TYPE_CONTROL, TOKEN_TYPE_NORMAL, TOKEN_TYPE_UNKNOWN, TOKEN_TYPE_UNUSED,
    TOKEN_TYPE_USER_DEFINED,
};

use ntd_core::ReasoningBudget;
use ntd_ir::IrVersion;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CognitiveSignals {
    pub complexity: f32,
    pub uncertainty: f32,
    pub prior_failure: bool,
}

impl CognitiveSignals {
    pub fn normalized(self) -> Self {
        Self {
            complexity: self.complexity.clamp(0.0, 1.0),
            uncertainty: self.uncertainty.clamp(0.0, 1.0),
            prior_failure: self.prior_failure,
        }
    }
}

pub fn accepts_ir(version: IrVersion) -> bool {
    IrVersion::CURRENT.can_read(version)
}

pub fn choose_reasoning_budget(signals: CognitiveSignals) -> ReasoningBudget {
    let s = signals.normalized();

    if s.prior_failure {
        ReasoningBudget::Recovery
    } else if s.complexity >= 0.70 || s.uncertainty >= 0.65 {
        ReasoningBudget::Deep
    } else if s.complexity >= 0.30 || s.uncertainty >= 0.25 {
        ReasoningBudget::Standard
    } else {
        ReasoningBudget::Reflex
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_accepts_current_ir_contract() {
        assert!(accepts_ir(IrVersion::CURRENT));
        assert!(!accepts_ir(IrVersion { major: 1, minor: 0 }));
    }

    #[test]
    fn simple_work_uses_reflex_budget() {
        assert_eq!(
            choose_reasoning_budget(CognitiveSignals {
                complexity: 0.1,
                uncertainty: 0.1,
                prior_failure: false,
            }),
            ReasoningBudget::Reflex
        );
    }

    #[test]
    fn failure_escalates_to_recovery() {
        assert_eq!(
            choose_reasoning_budget(CognitiveSignals {
                complexity: 0.0,
                uncertainty: 0.0,
                prior_failure: true,
            }),
            ReasoningBudget::Recovery
        );
    }
}
