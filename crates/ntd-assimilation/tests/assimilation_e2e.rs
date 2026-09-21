use ntd_assimilation::{
    AssetKind, AssimilationError, AssimilationIdentity, CapabilityForge, Discovery, ForgePolicy,
    ImporterRegistry, LicenseRecord, NativeAdapter, NativeAssetStore, NativeCandidate,
    NativeSection, NativeValidationSandbox, RegressionCase, RegressionProbe, SourceImporter,
    SourcePackage,
};
use ntd_capsule::{decode_graph_section, sha256, CapsuleView, SectionKind};
use ntd_core::{CapabilityId, SideEffectClass};
use ntd_ir::{
    DType, Graph, IrVersion, Node, NodeId, OpKind, TensorOp, ValueDecl, ValueId, ValueType,
};
use ntd_runtime::{CapabilityDescriptor, CapabilityDomain};

struct CapabilityImporter;

impl SourceImporter for CapabilityImporter {
    fn id(&self) -> &str {
        "test.capability.importer"
    }

    fn media_types(&self) -> &[&str] {
        &["application/x-test-capability"]
    }

    fn discover(&self, source: &SourcePackage) -> Result<Discovery, AssimilationError> {
        if source.payload.is_empty() {
            return Err(AssimilationError::InvalidSource);
        }
        Ok(Discovery {
            importer_id: self.id().into(),
            asset_kind: AssetKind::Capability,
            asset_id_hint: "skill.calculator".into(),
        })
    }

    fn import(&self, source: &SourcePackage) -> Result<NativeCandidate, AssimilationError> {
        let version = u32::from(source.payload[0]);
        let adapter_bytes = source.payload[1..].to_vec();
        let descriptor = CapabilityDescriptor::new(
            CapabilityId("custom.calculator".into()),
            version,
            CapabilityDomain::Custom,
            SideEffectClass::ReadOnly,
        )
        .map_err(|error| AssimilationError::ImportRejected(format!("{error:?}")))?;
        Ok(NativeCandidate::Capability {
            asset_id: "skill.calculator".into(),
            version,
            descriptor,
            adapter: NativeAdapter {
                format: "ntd97.adapter.declarative.v1".into(),
                bytes: adapter_bytes.clone(),
            },
            regressions: vec![RegressionCase {
                name: "adapter-integrity".into(),
                probe: RegressionProbe::AdapterDigest(sha256(&adapter_bytes)),
            }],
        })
    }
}

struct IntelligenceImporter;

impl SourceImporter for IntelligenceImporter {
    fn id(&self) -> &str {
        "test.intelligence.importer"
    }

    fn media_types(&self) -> &[&str] {
        &["application/x-external-add-graph"]
    }

    fn discover(&self, _source: &SourcePackage) -> Result<Discovery, AssimilationError> {
        Ok(Discovery {
            importer_id: self.id().into(),
            asset_kind: AssetKind::Intelligence,
            asset_id_hint: "intelligence.external-add".into(),
        })
    }

    fn import(&self, _source: &SourcePackage) -> Result<NativeCandidate, AssimilationError> {
        let scalar = |id| ValueDecl {
            id: ValueId(id),
            ty: ValueType::Scalar(DType::F32),
        };
        let graph = Graph {
            version: IrVersion::CURRENT,
            inputs: vec![scalar(0), scalar(1)],
            outputs: vec![ValueId(2)],
            nodes: vec![Node {
                id: NodeId(0),
                op: OpKind::Tensor(TensorOp::Add),
                inputs: vec![ValueId(0), ValueId(1)],
                outputs: vec![scalar(2)],
            }],
        };
        Ok(NativeCandidate::Intelligence {
            asset_id: "intelligence.external-add".into(),
            version: 1,
            graph,
            sections: vec![NativeSection {
                kind: SectionKind::Router,
                bytes: b"native-router".to_vec(),
            }],
            regressions: vec![RegressionCase {
                name: "nir97-roundtrip".into(),
                probe: RegressionProbe::GraphRoundTrip,
            }],
        })
    }
}

fn source(media_type: &str, bytes: Vec<u8>) -> SourcePackage {
    SourcePackage::new(
        media_type,
        bytes,
        "memory://fixture",
        LicenseRecord::new("MIT", "fixture license").expect("license"),
        "test fixture",
    )
    .expect("source")
}

fn forge() -> CapabilityForge<NativeValidationSandbox> {
    let mut importers = ImporterRegistry::new();
    importers
        .register(CapabilityImporter)
        .expect("capability importer");
    importers
        .register(IntelligenceImporter)
        .expect("intelligence importer");
    CapabilityForge::new(
        importers,
        NativeValidationSandbox,
        AssimilationIdentity::from_seed([91; 32]),
        ForgePolicy::new(vec!["MIT".into()], 1024 * 1024).expect("policy"),
    )
}

#[test]
fn capability_becomes_signed_native_capsule_without_source_runtime() {
    let mut forge = forge();
    let key = forge.identity().verify_key();
    let mut store = NativeAssetStore::new(key);
    let source = source(
        "application/x-test-capability",
        [vec![1], b"declarative-calculator-adapter".to_vec()].concat(),
    );

    let discovery = forge.discover(&source).expect("discover");
    assert_eq!(discovery.asset_kind, AssetKind::Capability);
    let receipt = forge.assimilate(&source, &mut store).expect("assimilate");
    assert_eq!(receipt.asset_id, "skill.calculator");

    let active = store.active("skill.calculator").expect("active");
    assert_eq!(active.version, 1);
    assert_ne!(active.native_capsule, source.payload);
    let descriptor =
        ntd_assimilation::load_native_capability(&active.native_capsule).expect("capability");
    assert_eq!(descriptor.id.0, "custom.calculator");

    let view = CapsuleView::read(&active.native_capsule).expect("capsule");
    let kinds = view
        .chunks
        .iter()
        .map(|chunk| chunk.kind)
        .collect::<Vec<_>>();
    assert!(kinds.contains(&SectionKind::Capabilities));
    assert!(kinds.contains(&SectionKind::Adapters));
    assert!(kinds.contains(&SectionKind::Provenance));
    assert!(kinds.contains(&SectionKind::Signatures));
    assert!(kinds.contains(&SectionKind::AssimilationLog));
}

#[test]
fn external_intelligence_normalizes_to_ntd97_ir_and_ncc97() {
    let mut forge = forge();
    let mut store = NativeAssetStore::new(forge.identity().verify_key());
    let source = source(
        "application/x-external-add-graph",
        b"external-source-graph:add".to_vec(),
    );

    forge.assimilate(&source, &mut store).expect("assimilate");
    let active = store.active("intelligence.external-add").expect("active");
    let view = CapsuleView::read(&active.native_capsule).expect("capsule");
    let graph_chunk = view
        .chunks
        .iter()
        .find(|chunk| chunk.kind == SectionKind::Graph)
        .expect("graph");
    let graph = decode_graph_section(graph_chunk).expect("decode graph");
    assert_eq!(graph.nodes.len(), 1);
    assert!(matches!(graph.nodes[0].op, OpKind::Tensor(TensorOp::Add)));
}

#[test]
fn versions_commit_monotonically_and_can_rollback() {
    let mut forge = forge();
    let mut store = NativeAssetStore::new(forge.identity().verify_key());

    forge
        .assimilate(
            &source(
                "application/x-test-capability",
                [vec![1], b"adapter-v1".to_vec()].concat(),
            ),
            &mut store,
        )
        .expect("v1");
    forge
        .assimilate(
            &source(
                "application/x-test-capability",
                [vec![2], b"adapter-v2".to_vec()].concat(),
            ),
            &mut store,
        )
        .expect("v2");

    assert_eq!(store.versions("skill.calculator"), vec![1, 2]);
    assert_eq!(store.active("skill.calculator").expect("active").version, 2);
    store.rollback("skill.calculator", 1).expect("rollback");
    assert_eq!(store.active("skill.calculator").expect("active").version, 1);
}

#[test]
fn license_denial_and_failed_batch_leave_store_unchanged() {
    let mut forge = forge();
    let mut store = NativeAssetStore::new(forge.identity().verify_key());
    let denied = SourcePackage::new(
        "application/x-test-capability",
        [vec![1], b"adapter".to_vec()].concat(),
        "memory://denied",
        LicenseRecord::new("LicenseRef-Prohibited", "").expect("license"),
        "",
    )
    .expect("source");
    assert!(matches!(
        forge.assimilate(&denied, &mut store),
        Err(AssimilationError::LicenseDenied(_))
    ));
    assert_eq!(store.asset_count(), 0);

    let good_source = source(
        "application/x-test-capability",
        [vec![1], b"adapter".to_vec()].concat(),
    );
    forge
        .assimilate(&good_source, &mut store)
        .expect("first commit");
    let before = store.asset_count();
    let duplicate = forge.assimilate(&good_source, &mut store);
    assert_eq!(duplicate, Err(AssimilationError::VersionConflict));
    assert_eq!(store.asset_count(), before);
    assert_eq!(store.active("skill.calculator").expect("active").version, 1);
}

#[test]
fn package_signature_rejects_tampering_and_untrusted_signer() {
    let mut forge = forge();
    let mut store = NativeAssetStore::new(forge.identity().verify_key());
    forge
        .assimilate(
            &source(
                "application/x-test-capability",
                [vec![1], b"adapter".to_vec()].concat(),
            ),
            &mut store,
        )
        .expect("commit");
    let package = store.active("skill.calculator").expect("active").clone();

    assert_eq!(
        ntd_assimilation::verify_native_package(&package, &[7; 32]),
        Err(AssimilationError::UntrustedSigner)
    );

    let mut tampered = package;
    let last = tampered.native_capsule.len() - 1;
    tampered.native_capsule[last] ^= 1;
    assert!(
        ntd_assimilation::verify_native_package(&tampered, &forge.identity().verify_key()).is_err()
    );
}
