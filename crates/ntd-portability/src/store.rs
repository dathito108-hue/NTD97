#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use ntd_capsule::{sha256, Digest};

use crate::{
    crypto::{object_aad, object_nonce},
    BackupKey, PortabilityError,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncryptedObject {
    pub digest: Digest,
    pub logical_len: u64,
    pub ciphertext: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct SovereignObjectStore {
    key: BackupKey,
    objects: BTreeMap<Digest, EncryptedObject>,
}

impl SovereignObjectStore {
    pub fn new(key: BackupKey) -> Self {
        Self {
            key,
            objects: BTreeMap::new(),
        }
    }

    pub fn insert(&mut self, bytes: &[u8]) -> Result<Digest, PortabilityError> {
        let digest = sha256(bytes);
        let logical_len = u64::try_from(bytes.len()).map_err(|_| PortabilityError::Overflow)?;
        if self.objects.contains_key(&digest) {
            let restored = self.get(&digest)?;
            if restored != bytes {
                return Err(PortabilityError::IntegrityMismatch);
            }
            return Ok(digest);
        }
        let ciphertext = self.key.seal(
            object_nonce(&digest),
            &object_aad(&digest, logical_len),
            bytes,
        )?;
        self.objects.insert(
            digest,
            EncryptedObject {
                digest,
                logical_len,
                ciphertext,
            },
        );
        Ok(digest)
    }

    pub fn import(&mut self, object: EncryptedObject) -> Result<(), PortabilityError> {
        let plaintext = self.decrypt_record(&object)?;
        if sha256(&plaintext) != object.digest
            || u64::try_from(plaintext.len()).map_err(|_| PortabilityError::Overflow)?
                != object.logical_len
        {
            return Err(PortabilityError::IntegrityMismatch);
        }
        if let Some(existing) = self.objects.get(&object.digest) {
            let existing_plaintext = self.decrypt_record(existing)?;
            if existing_plaintext != plaintext {
                return Err(PortabilityError::IntegrityMismatch);
            }
            return Ok(());
        }
        self.objects.insert(object.digest, object);
        Ok(())
    }

    pub fn get(&self, digest: &Digest) -> Result<Vec<u8>, PortabilityError> {
        let object = self
            .objects
            .get(digest)
            .ok_or(PortabilityError::MissingObject(*digest))?;
        self.decrypt_record(object)
    }

    pub fn record(&self, digest: &Digest) -> Result<EncryptedObject, PortabilityError> {
        self.objects
            .get(digest)
            .cloned()
            .ok_or(PortabilityError::MissingObject(*digest))
    }

    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    pub(crate) fn seal_manifest(
        &self,
        nonce: [u8; 24],
        aad: &[u8],
        plaintext: &[u8],
    ) -> Result<Vec<u8>, PortabilityError> {
        self.key.seal(nonce, aad, plaintext)
    }

    pub(crate) fn open_manifest(
        &self,
        nonce: [u8; 24],
        aad: &[u8],
        ciphertext: &[u8],
    ) -> Result<Vec<u8>, PortabilityError> {
        self.key.open(nonce, aad, ciphertext)
    }

    fn decrypt_record(&self, object: &EncryptedObject) -> Result<Vec<u8>, PortabilityError> {
        let plaintext = self.key.open(
            object_nonce(&object.digest),
            &object_aad(&object.digest, object.logical_len),
            &object.ciphertext,
        )?;
        if sha256(&plaintext) != object.digest {
            return Err(PortabilityError::IntegrityMismatch);
        }
        Ok(plaintext)
    }
}
