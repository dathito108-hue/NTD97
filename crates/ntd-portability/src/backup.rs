#![forbid(unsafe_code)]

use std::collections::BTreeSet;

use ntd_capsule::Digest;

use crate::{
    AssetClass, EncryptedObject, PortabilityError, PortableAsset, RestoredSet, SovereignObjectStore,
};

const BACKUP_MAGIC: [u8; 6] = *b"PSB97\0";
const MANIFEST_MAGIC: [u8; 6] = *b"PSM97\0";
const BACKUP_MAJOR: u16 = 0;
const BACKUP_MINOR: u16 = 1;
const BACKUP_HEADER_LEN: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BackupKind {
    Full = 1,
    Thin = 2,
    State = 3,
}

impl TryFrom<u8> for BackupKind {
    type Error = PortabilityError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Full),
            2 => Ok(Self::Thin),
            3 => Ok(Self::State),
            _ => Err(PortabilityError::InvalidBackup),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortableBackup {
    pub backup_id: [u8; 16],
    pub kind: BackupKind,
    pub manifest_nonce: [u8; 24],
    pub encrypted_manifest: Vec<u8>,
    pub objects: Vec<EncryptedObject>,
}

impl PortableBackup {
    pub fn encode(&self) -> Result<Vec<u8>, PortabilityError> {
        if self.encrypted_manifest.is_empty() {
            return Err(PortabilityError::InvalidBackup);
        }
        let mut out = Vec::new();
        out.extend_from_slice(&BACKUP_MAGIC);
        push_u16(&mut out, BACKUP_MAJOR);
        push_u16(&mut out, BACKUP_MINOR);
        out.push(self.kind as u8);
        out.push(0);
        out.extend_from_slice(&self.backup_id);
        out.extend_from_slice(&self.manifest_nonce);
        push_u64(
            &mut out,
            u64::try_from(self.encrypted_manifest.len()).map_err(|_| PortabilityError::Overflow)?,
        );
        push_u32(
            &mut out,
            u32::try_from(self.objects.len()).map_err(|_| PortabilityError::Overflow)?,
        );
        if out.len() != BACKUP_HEADER_LEN {
            return Err(PortabilityError::InvalidBackup);
        }
        out.extend_from_slice(&self.encrypted_manifest);
        for object in &self.objects {
            out.extend_from_slice(&object.digest);
            push_u64(&mut out, object.logical_len);
            push_u64(
                &mut out,
                u64::try_from(object.ciphertext.len()).map_err(|_| PortabilityError::Overflow)?,
            );
            out.extend_from_slice(&object.ciphertext);
        }
        Ok(out)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, PortabilityError> {
        let mut cursor = Cursor::new(bytes);
        if cursor.take(6)? != BACKUP_MAGIC
            || cursor.u16()? != BACKUP_MAJOR
            || cursor.u16()? > BACKUP_MINOR
        {
            return Err(PortabilityError::InvalidBackup);
        }
        let kind = BackupKind::try_from(cursor.u8()?)?;
        if cursor.u8()? != 0 {
            return Err(PortabilityError::InvalidBackup);
        }
        let mut backup_id = [0u8; 16];
        backup_id.copy_from_slice(cursor.take(16)?);
        let mut manifest_nonce = [0u8; 24];
        manifest_nonce.copy_from_slice(cursor.take(24)?);
        let manifest_len =
            usize::try_from(cursor.u64()?).map_err(|_| PortabilityError::Overflow)?;
        let object_count =
            usize::try_from(cursor.u32()?).map_err(|_| PortabilityError::Overflow)?;
        let encrypted_manifest = cursor.take(manifest_len)?.to_vec();
        let mut objects = Vec::with_capacity(object_count);
        let mut previous: Option<Digest> = None;
        for _ in 0..object_count {
            let mut digest = [0u8; 32];
            digest.copy_from_slice(cursor.take(32)?);
            if previous.is_some_and(|value| digest <= value) {
                return Err(PortabilityError::InvalidBackup);
            }
            previous = Some(digest);
            let logical_len = cursor.u64()?;
            let cipher_len =
                usize::try_from(cursor.u64()?).map_err(|_| PortabilityError::Overflow)?;
            objects.push(EncryptedObject {
                digest,
                logical_len,
                ciphertext: cursor.take(cipher_len)?.to_vec(),
            });
        }
        if encrypted_manifest.is_empty() || !cursor.is_finished() {
            return Err(PortabilityError::InvalidBackup);
        }
        Ok(Self {
            backup_id,
            kind,
            manifest_nonce,
            encrypted_manifest,
            objects,
        })
    }
}

#[derive(Debug, Clone)]
struct ManifestAsset {
    class: AssetClass,
    asset_id: String,
    version: u32,
    logical_len: u64,
    digest: Digest,
}

pub struct PortableBackupBuilder;

impl PortableBackupBuilder {
    pub fn create(
        kind: BackupKind,
        backup_id: [u8; 16],
        manifest_nonce: [u8; 24],
        assets: &[PortableAsset],
        store: &mut SovereignObjectStore,
    ) -> Result<PortableBackup, PortabilityError> {
        let selected = assets
            .iter()
            .filter(|asset| kind != BackupKind::State || asset.is_stateful())
            .collect::<Vec<_>>();
        if selected.is_empty() {
            return Err(PortabilityError::EmptyBackup);
        }

        let mut seen = BTreeSet::new();
        let mut manifest_assets = Vec::with_capacity(selected.len());
        for asset in selected {
            let identity = (asset.class, asset.asset_id.clone(), asset.version);
            if !seen.insert(identity) {
                return Err(PortabilityError::DuplicateAsset);
            }
            let digest = store.insert(&asset.capsule)?;
            if digest != asset.digest {
                return Err(PortabilityError::IntegrityMismatch);
            }
            manifest_assets.push(ManifestAsset {
                class: asset.class,
                asset_id: asset.asset_id.clone(),
                version: asset.version,
                logical_len: u64::try_from(asset.capsule.len())
                    .map_err(|_| PortabilityError::Overflow)?,
                digest,
            });
        }
        manifest_assets.sort_by(|a, b| {
            (a.class, a.asset_id.as_str(), a.version).cmp(&(
                b.class,
                b.asset_id.as_str(),
                b.version,
            ))
        });

        let manifest = encode_manifest(kind, backup_id, &manifest_assets)?;
        let aad = manifest_aad(kind, backup_id);
        let encrypted_manifest = store.seal_manifest(manifest_nonce, &aad, &manifest)?;

        let objects = if kind == BackupKind::Thin {
            Vec::new()
        } else {
            let mut digests = manifest_assets
                .iter()
                .map(|asset| asset.digest)
                .collect::<Vec<_>>();
            digests.sort();
            digests.dedup();
            let mut objects = Vec::with_capacity(digests.len());
            for digest in digests {
                objects.push(store.record(&digest)?);
            }
            objects
        };

        Ok(PortableBackup {
            backup_id,
            kind,
            manifest_nonce,
            encrypted_manifest,
            objects,
        })
    }
}

pub fn restore_backup(
    backup: &PortableBackup,
    store: &mut SovereignObjectStore,
) -> Result<RestoredSet, PortabilityError> {
    let mut staged = store.clone();
    for object in &backup.objects {
        staged.import(object.clone())?;
    }

    let aad = manifest_aad(backup.kind, backup.backup_id);
    let manifest = staged.open_manifest(backup.manifest_nonce, &aad, &backup.encrypted_manifest)?;
    let entries = decode_manifest(&manifest, backup.kind, backup.backup_id)?;

    let mut assets = Vec::with_capacity(entries.len());
    for entry in entries {
        let capsule = staged.get(&entry.digest)?;
        if u64::try_from(capsule.len()).map_err(|_| PortabilityError::Overflow)?
            != entry.logical_len
        {
            return Err(PortabilityError::IntegrityMismatch);
        }
        let asset =
            PortableAsset::from_capsule(entry.class, entry.asset_id, entry.version, capsule)?;
        if asset.digest != entry.digest {
            return Err(PortabilityError::IntegrityMismatch);
        }
        assets.push(asset);
    }

    *store = staged;
    Ok(RestoredSet { assets })
}

fn encode_manifest(
    kind: BackupKind,
    backup_id: [u8; 16],
    assets: &[ManifestAsset],
) -> Result<Vec<u8>, PortabilityError> {
    let mut out = Vec::new();
    out.extend_from_slice(&MANIFEST_MAGIC);
    push_u16(&mut out, BACKUP_MAJOR);
    push_u16(&mut out, BACKUP_MINOR);
    out.push(kind as u8);
    out.push(0);
    out.extend_from_slice(&backup_id);
    push_u32(
        &mut out,
        u32::try_from(assets.len()).map_err(|_| PortabilityError::Overflow)?,
    );
    for asset in assets {
        out.push(asset.class as u8);
        push_u32(&mut out, asset.version);
        push_string(&mut out, &asset.asset_id)?;
        push_u64(&mut out, asset.logical_len);
        out.extend_from_slice(&asset.digest);
    }
    Ok(out)
}

fn decode_manifest(
    bytes: &[u8],
    expected_kind: BackupKind,
    expected_id: [u8; 16],
) -> Result<Vec<ManifestAsset>, PortabilityError> {
    let mut cursor = Cursor::new(bytes);
    if cursor.take(6)? != MANIFEST_MAGIC
        || cursor.u16()? != BACKUP_MAJOR
        || cursor.u16()? > BACKUP_MINOR
        || BackupKind::try_from(cursor.u8()?)? != expected_kind
        || cursor.u8()? != 0
    {
        return Err(PortabilityError::InvalidBackup);
    }
    let mut backup_id = [0u8; 16];
    backup_id.copy_from_slice(cursor.take(16)?);
    if backup_id != expected_id {
        return Err(PortabilityError::InvalidBackup);
    }
    let count = usize::try_from(cursor.u32()?).map_err(|_| PortabilityError::Overflow)?;
    let mut assets = Vec::with_capacity(count);
    let mut previous: Option<(AssetClass, String, u32)> = None;
    for _ in 0..count {
        let class = AssetClass::try_from(cursor.u8()?)?;
        let version = cursor.u32()?;
        let asset_id = cursor.string()?;
        if version == 0 || asset_id.trim().is_empty() {
            return Err(PortabilityError::InvalidBackup);
        }
        let identity = (class, asset_id.clone(), version);
        if previous.as_ref().is_some_and(|value| identity <= *value) {
            return Err(PortabilityError::InvalidBackup);
        }
        previous = Some(identity);
        let logical_len = cursor.u64()?;
        let mut digest = [0u8; 32];
        digest.copy_from_slice(cursor.take(32)?);
        assets.push(ManifestAsset {
            class,
            asset_id,
            version,
            logical_len,
            digest,
        });
    }
    if assets.is_empty() || !cursor.is_finished() {
        return Err(PortabilityError::InvalidBackup);
    }
    Ok(assets)
}

fn manifest_aad(kind: BackupKind, backup_id: [u8; 16]) -> Vec<u8> {
    let mut aad = b"NTD97-PSI-MANIFEST-v1".to_vec();
    aad.push(kind as u8);
    aad.extend_from_slice(&backup_id);
    aad
}

fn push_string(out: &mut Vec<u8>, value: &str) -> Result<(), PortabilityError> {
    let bytes = value.as_bytes();
    push_u32(
        out,
        u32::try_from(bytes.len()).map_err(|_| PortabilityError::Overflow)?,
    );
    out.extend_from_slice(bytes);
    Ok(())
}

fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], PortabilityError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(PortabilityError::Overflow)?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or(PortabilityError::InvalidBackup)?;
        self.offset = end;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8, PortabilityError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, PortabilityError> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self) -> Result<u32, PortabilityError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn u64(&mut self) -> Result<u64, PortabilityError> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn string(&mut self) -> Result<String, PortabilityError> {
        let len = usize::try_from(self.u32()?).map_err(|_| PortabilityError::Overflow)?;
        String::from_utf8(self.take(len)?.to_vec()).map_err(|_| PortabilityError::InvalidBackup)
    }

    fn is_finished(&self) -> bool {
        self.offset == self.bytes.len()
    }
}
