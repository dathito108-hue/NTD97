use ntd_capsule::{CapsuleBuilder, CapsuleKind, SectionKind};
use ntd_portability::{
    restore_backup, AssetClass, BackupKey, BackupKind, PortabilityError, PortableAsset,
    PortableBackup, PortableBackupBuilder, SovereignObjectStore,
};

fn full_capsule(tag: &[u8]) -> Vec<u8> {
    let mut builder = CapsuleBuilder::new(CapsuleKind::Full, [1; 16]);
    builder.push_embedded(SectionKind::Router, tag.to_vec());
    builder.write().expect("full capsule")
}

fn state_capsule(tag: &[u8]) -> Vec<u8> {
    let mut builder = CapsuleBuilder::new(CapsuleKind::State, [2; 16]);
    builder.push_embedded(SectionKind::MemoryState, tag.to_vec());
    builder.write().expect("state capsule")
}

fn assets() -> Vec<PortableAsset> {
    vec![
        PortableAsset::from_capsule(AssetClass::Model, "model.core", 3, full_capsule(b"model"))
            .expect("model"),
        PortableAsset::from_capsule(
            AssetClass::Capability,
            "cap.web",
            2,
            full_capsule(b"capability"),
        )
        .expect("capability"),
        PortableAsset::from_capsule(
            AssetClass::Memory,
            "memory.user",
            7,
            state_capsule(b"memory"),
        )
        .expect("memory"),
        PortableAsset::from_capsule(
            AssetClass::State,
            "state.identity",
            11,
            state_capsule(b"identity-state"),
        )
        .expect("state"),
    ]
}

fn key() -> BackupKey {
    BackupKey::from_bytes([71; 32])
}

#[test]
fn full_backup_round_trips_offline_on_another_device() {
    let assets = assets();
    let mut source_store = SovereignObjectStore::new(key());
    let backup = PortableBackupBuilder::create(
        BackupKind::Full,
        [9; 16],
        [10; 24],
        &assets,
        &mut source_store,
    )
    .expect("backup");
    let encoded = backup.encode().expect("encode");
    let decoded = PortableBackup::decode(&encoded).expect("decode");

    let mut destination_store = SovereignObjectStore::new(key());
    let restored = restore_backup(&decoded, &mut destination_store).expect("restore");
    assert_eq!(restored.assets.len(), 4);
    assert!(restored.asset(AssetClass::Model, "model.core").is_some());
    assert!(restored.asset(AssetClass::Capability, "cap.web").is_some());
    assert!(restored.asset(AssetClass::Memory, "memory.user").is_some());
    assert!(restored
        .asset(AssetClass::State, "state.identity")
        .is_some());
}

#[test]
fn thin_backup_uses_content_addressed_dedup_store() {
    let assets = assets();
    let mut source_store = SovereignObjectStore::new(key());
    let full = PortableBackupBuilder::create(
        BackupKind::Full,
        [11; 16],
        [12; 24],
        &assets,
        &mut source_store,
    )
    .expect("full");
    let before = source_store.object_count();
    let thin = PortableBackupBuilder::create(
        BackupKind::Thin,
        [13; 16],
        [14; 24],
        &assets,
        &mut source_store,
    )
    .expect("thin");
    assert_eq!(source_store.object_count(), before);
    assert!(thin.objects.is_empty());

    let mut destination_store = SovereignObjectStore::new(key());
    for object in full.objects {
        destination_store.import(object).expect("seed object");
    }
    let restored = restore_backup(&thin, &mut destination_store).expect("thin restore");
    assert_eq!(restored.assets.len(), 4);
}

#[test]
fn state_backup_excludes_model_and_capability_assets() {
    let assets = assets();
    let mut store = SovereignObjectStore::new(key());
    let backup =
        PortableBackupBuilder::create(BackupKind::State, [15; 16], [16; 24], &assets, &mut store)
            .expect("state backup");
    let mut destination = SovereignObjectStore::new(key());
    let restored = restore_backup(&backup, &mut destination).expect("restore");
    assert_eq!(restored.assets.len(), 2);
    assert!(restored.asset(AssetClass::Model, "model.core").is_none());
    assert!(restored.asset(AssetClass::Capability, "cap.web").is_none());
    assert!(restored.asset(AssetClass::Memory, "memory.user").is_some());
    assert!(restored
        .asset(AssetClass::State, "state.identity")
        .is_some());
}

#[test]
fn wrong_user_key_cannot_restore_backup() {
    let assets = assets();
    let mut source_store = SovereignObjectStore::new(key());
    let backup = PortableBackupBuilder::create(
        BackupKind::Full,
        [17; 16],
        [18; 24],
        &assets,
        &mut source_store,
    )
    .expect("backup");

    let mut wrong_store = SovereignObjectStore::new(BackupKey::from_bytes([72; 32]));
    assert_eq!(
        restore_backup(&backup, &mut wrong_store),
        Err(PortabilityError::CryptoFailure)
    );
}

#[test]
fn corrupted_object_is_rejected_during_sovereign_recovery() {
    let assets = assets();
    let mut source_store = SovereignObjectStore::new(key());
    let mut backup = PortableBackupBuilder::create(
        BackupKind::Full,
        [19; 16],
        [20; 24],
        &assets,
        &mut source_store,
    )
    .expect("backup");
    backup.objects[0].ciphertext[0] ^= 1;

    let mut destination = SovereignObjectStore::new(key());
    assert!(restore_backup(&backup, &mut destination).is_err());
    assert_eq!(destination.object_count(), 0);
}

#[test]
fn thin_restore_fails_closed_when_dedup_object_is_missing() {
    let assets = assets();
    let mut source_store = SovereignObjectStore::new(key());
    let thin = PortableBackupBuilder::create(
        BackupKind::Thin,
        [21; 16],
        [22; 24],
        &assets,
        &mut source_store,
    )
    .expect("thin");

    let mut empty_store = SovereignObjectStore::new(key());
    assert!(matches!(
        restore_backup(&thin, &mut empty_store),
        Err(PortabilityError::MissingObject(_))
    ));
}
