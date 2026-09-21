#![forbid(unsafe_code)]

use ntd_capsule::{
    sha256, CapsuleBuilder, CapsuleKind, CapsuleView, ChunkStorageView, SectionKind,
};
use ntd_core::{Intent, TaskGraph};
use ntd_runtime::{
    decode_cognitive_checkpoint, encode_cognitive_checkpoint, CognitiveIdentity, CognitiveRuntime,
    MemoryKind,
};

#[test]
fn state_capsule_restores_sovereign_cognitive_identity() {
    let identity = CognitiveIdentity(*b"NTD97-COGNITION1");
    let mut runtime = CognitiveRuntime::new(identity);
    let goal = runtime
        .state_mut()
        .add_goal("survive process death", vec!["restore state".into()])
        .expect("goal");
    let task = runtime
        .state_mut()
        .submit_task(
            Intent::new("resume after cold start"),
            TaskGraph::default(),
            Some(goal),
        )
        .expect("task");
    runtime
        .state_mut()
        .set_world_fact("continuity", "checkpointed")
        .expect("world");
    let memory_id = runtime
        .state_mut()
        .memory
        .store(
            MemoryKind::Procedural,
            "restore SIK97 before resuming task execution",
            vec!["continuity".into()],
            950,
            0,
        )
        .expect("memory");

    let checkpoint = encode_cognitive_checkpoint(runtime.state()).expect("encode checkpoint");

    let base_root = sha256(b"native-generative-base");
    let mut builder =
        CapsuleBuilder::new(CapsuleKind::State, *b"NTD97-STATE-0001").with_base_root(base_root);
    builder.push_embedded(SectionKind::ContinuityState, checkpoint.clone());

    let bytes = builder.write().expect("write state capsule");
    let view = CapsuleView::read(&bytes).expect("read state capsule");

    assert_eq!(view.kind, CapsuleKind::State);
    assert_eq!(view.base_root, Some(base_root));

    let continuity = view
        .chunks
        .iter()
        .find(|chunk| chunk.kind == SectionKind::ContinuityState)
        .expect("continuity chunk");

    let stored = match continuity.storage {
        ChunkStorageView::Embedded(bytes) => bytes,
        ChunkStorageView::External => panic!("state checkpoint must be embedded in this test"),
    };
    assert_eq!(stored, checkpoint.as_slice());

    let state = decode_cognitive_checkpoint(stored).expect("decode restored state");
    let restored = CognitiveRuntime::from_state(state).expect("restore runtime");

    assert_eq!(restored.state().identity, identity);
    assert!(restored.state().tasks.contains_key(&task));
    assert_eq!(
        restored
            .state()
            .world
            .get("continuity")
            .expect("world")
            .value,
        "checkpointed"
    );
    assert_eq!(
        restored.state().memory.get(memory_id).expect("memory").kind,
        MemoryKind::Procedural
    );
}
