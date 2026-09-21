#![forbid(unsafe_code)]

use std::fmt;

use chacha20poly1305::{
    aead::{Aead, Payload},
    KeyInit, XChaCha20Poly1305, XNonce,
};
use ntd_capsule::{sha256, Digest};
use zeroize::Zeroize;

use crate::PortabilityError;

#[derive(Clone, PartialEq, Eq)]
pub struct BackupKey([u8; 32]);

impl BackupKey {
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub(crate) fn seal(
        &self,
        nonce: [u8; 24],
        aad: &[u8],
        plaintext: &[u8],
    ) -> Result<Vec<u8>, PortabilityError> {
        let cipher = XChaCha20Poly1305::new((&self.0).into());
        cipher
            .encrypt(
                XNonce::from_slice(&nonce),
                Payload {
                    msg: plaintext,
                    aad,
                },
            )
            .map_err(|_| PortabilityError::CryptoFailure)
    }

    pub(crate) fn open(
        &self,
        nonce: [u8; 24],
        aad: &[u8],
        ciphertext: &[u8],
    ) -> Result<Vec<u8>, PortabilityError> {
        let cipher = XChaCha20Poly1305::new((&self.0).into());
        cipher
            .decrypt(
                XNonce::from_slice(&nonce),
                Payload {
                    msg: ciphertext,
                    aad,
                },
            )
            .map_err(|_| PortabilityError::CryptoFailure)
    }
}

impl fmt::Debug for BackupKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("BackupKey(<redacted>)")
    }
}

impl Drop for BackupKey {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

pub(crate) fn object_nonce(digest: &Digest) -> [u8; 24] {
    let mut material = b"NTD97-PSI-OBJECT-NONCE-v1".to_vec();
    material.extend_from_slice(digest);
    let derived = sha256(&material);
    let mut nonce = [0u8; 24];
    nonce.copy_from_slice(&derived[..24]);
    nonce
}

pub(crate) fn object_aad(digest: &Digest, logical_len: u64) -> Vec<u8> {
    let mut aad = b"NTD97-PSI-OBJECT-v1".to_vec();
    aad.extend_from_slice(digest);
    aad.extend_from_slice(&logical_len.to_le_bytes());
    aad
}
