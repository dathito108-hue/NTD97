#![forbid(unsafe_code)]

use std::fmt;

use chacha20poly1305::{
    aead::{Aead, Payload},
    ChaCha20Poly1305, KeyInit, Nonce,
};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use hkdf::Hkdf;
use sha2::{Digest, Sha256};
use x25519_dalek::{PublicKey as X25519PublicKey, StaticSecret};

use crate::PcFabricError;

const HANDSHAKE_DOMAIN: &[u8] = b"NTD97-PCF97-HANDSHAKE-v1";
const SESSION_KDF_INFO: &[u8] = b"NTD97-PCF97-SESSION-v1";
const SESSION_FRAME_MAGIC: [u8; 6] = *b"PCS97\0";
const SESSION_FRAME_MAJOR: u16 = 0;
const SESSION_FRAME_MINOR: u16 = 1;
const SESSION_HEADER_LEN: usize = 38;

#[derive(Clone, PartialEq, Eq)]
pub struct PairedIdentity {
    signing_seed: [u8; 32],
    peer_id: [u8; 16],
    verify_key: [u8; 32],
}

impl fmt::Debug for PairedIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PairedIdentity")
            .field("peer_id", &self.peer_id)
            .field("verify_key", &self.verify_key)
            .field("signing_seed", &"<redacted>")
            .finish()
    }
}

impl PairedIdentity {
    pub fn from_seed(signing_seed: [u8; 32]) -> Self {
        let signing = SigningKey::from_bytes(&signing_seed);
        let verify_key = signing.verifying_key().to_bytes();
        let peer_id = peer_id_for_verify_key(&verify_key);

        Self {
            signing_seed,
            peer_id,
            verify_key,
        }
    }

    pub fn peer_id(&self) -> [u8; 16] {
        self.peer_id
    }

    pub fn verify_key(&self) -> [u8; 32] {
        self.verify_key
    }

    pub fn pairing_record(&self) -> PairingRecord {
        PairingRecord {
            remote_peer_id: self.peer_id,
            remote_verify_key: self.verify_key,
        }
    }

    fn sign(&self, bytes: &[u8]) -> [u8; 64] {
        SigningKey::from_bytes(&self.signing_seed)
            .sign(bytes)
            .to_bytes()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairingRecord {
    remote_peer_id: [u8; 16],
    remote_verify_key: [u8; 32],
}

impl PairingRecord {
    pub fn from_public(
        remote_peer_id: [u8; 16],
        remote_verify_key: [u8; 32],
    ) -> Result<Self, PcFabricError> {
        if peer_id_for_verify_key(&remote_verify_key) != remote_peer_id {
            return Err(PcFabricError::InvalidPairing);
        }
        VerifyingKey::from_bytes(&remote_verify_key).map_err(|_| PcFabricError::InvalidPairing)?;

        Ok(Self {
            remote_peer_id,
            remote_verify_key,
        })
    }

    pub fn remote_peer_id(&self) -> [u8; 16] {
        self.remote_peer_id
    }

    pub fn remote_verify_key(&self) -> [u8; 32] {
        self.remote_verify_key
    }

    fn verify(&self, bytes: &[u8], signature: &[u8; 64]) -> Result<(), PcFabricError> {
        let key = VerifyingKey::from_bytes(&self.remote_verify_key)
            .map_err(|_| PcFabricError::InvalidPairing)?;
        let signature = Signature::from_bytes(signature);
        key.verify(bytes, &signature)
            .map_err(|_| PcFabricError::InvalidSignature)
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct HandshakeEntropy([u8; 32]);

impl fmt::Debug for HandshakeEntropy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("HandshakeEntropy(<redacted>)")
    }
}

impl HandshakeEntropy {
    pub fn from_seed(seed: [u8; 32]) -> Self {
        Self(seed)
    }

    fn x25519_secret(&self) -> StaticSecret {
        StaticSecret::from(derive32(b"x25519", &self.0))
    }

    fn challenge(&self) -> [u8; 32] {
        derive32(b"challenge", &self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientHello {
    pub client_peer_id: [u8; 16],
    pub expected_server_peer_id: [u8; 16],
    pub session_id: [u8; 16],
    pub ephemeral_public: [u8; 32],
    pub challenge: [u8; 32],
    pub signature: [u8; 64],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerHello {
    pub server_peer_id: [u8; 16],
    pub expected_client_peer_id: [u8; 16],
    pub session_id: [u8; 16],
    pub ephemeral_public: [u8; 32],
    pub challenge: [u8; 32],
    pub client_hello_digest: [u8; 32],
    pub signature: [u8; 64],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionRole {
    Client,
    Server,
}

#[derive(Clone)]
pub struct ClientHandshake {
    local: PairedIdentity,
    remote: PairingRecord,
    entropy: HandshakeEntropy,
    hello: ClientHello,
}

impl fmt::Debug for ClientHandshake {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ClientHandshake")
            .field("local_peer_id", &self.local.peer_id)
            .field("remote_peer_id", &self.remote.remote_peer_id)
            .field("session_id", &self.hello.session_id)
            .finish()
    }
}

impl ClientHandshake {
    pub fn begin(
        local: PairedIdentity,
        remote: PairingRecord,
        entropy: HandshakeEntropy,
    ) -> Result<(Self, ClientHello), PcFabricError> {
        validate_identity_pair(&local, &remote)?;
        let secret = entropy.x25519_secret();
        let ephemeral_public = X25519PublicKey::from(&secret).to_bytes();
        let challenge = entropy.challenge();
        let session_id = session_id(
            &local.peer_id,
            &remote.remote_peer_id,
            &ephemeral_public,
            &challenge,
        );

        let mut hello = ClientHello {
            client_peer_id: local.peer_id,
            expected_server_peer_id: remote.remote_peer_id,
            session_id,
            ephemeral_public,
            challenge,
            signature: [0; 64],
        };
        hello.signature = local.sign(&client_signing_bytes(&hello));

        Ok((
            Self {
                local,
                remote,
                entropy,
                hello: hello.clone(),
            },
            hello,
        ))
    }

    pub fn finish(self, server: &ServerHello) -> Result<SecureSession, PcFabricError> {
        if server.server_peer_id != self.remote.remote_peer_id
            || server.expected_client_peer_id != self.local.peer_id
            || server.session_id != self.hello.session_id
            || server.client_hello_digest != digest(&client_wire_bytes(&self.hello))
        {
            return Err(PcFabricError::InvalidHandshake);
        }

        self.remote
            .verify(&server_signing_bytes(server), &server.signature)?;

        let secret = self.entropy.x25519_secret();
        let remote_public = X25519PublicKey::from(server.ephemeral_public);
        let shared = secret.diffie_hellman(&remote_public);
        derive_session(
            SessionRole::Client,
            self.hello.session_id,
            shared.as_bytes(),
            &self.hello,
            server,
        )
    }
}

impl ServerHello {
    pub fn accept(
        local: PairedIdentity,
        remote: PairingRecord,
        entropy: HandshakeEntropy,
        client: &ClientHello,
    ) -> Result<(Self, SecureSession), PcFabricError> {
        validate_identity_pair(&local, &remote)?;
        if client.client_peer_id != remote.remote_peer_id
            || client.expected_server_peer_id != local.peer_id
        {
            return Err(PcFabricError::PeerMismatch);
        }

        remote.verify(&client_signing_bytes(client), &client.signature)?;

        let expected_session_id = session_id(
            &client.client_peer_id,
            &client.expected_server_peer_id,
            &client.ephemeral_public,
            &client.challenge,
        );
        if client.session_id != expected_session_id {
            return Err(PcFabricError::InvalidHandshake);
        }

        let secret = entropy.x25519_secret();
        let ephemeral_public = X25519PublicKey::from(&secret).to_bytes();
        let challenge = entropy.challenge();

        let mut hello = Self {
            server_peer_id: local.peer_id,
            expected_client_peer_id: remote.remote_peer_id,
            session_id: client.session_id,
            ephemeral_public,
            challenge,
            client_hello_digest: digest(&client_wire_bytes(client)),
            signature: [0; 64],
        };
        hello.signature = local.sign(&server_signing_bytes(&hello));

        let remote_public = X25519PublicKey::from(client.ephemeral_public);
        let shared = secret.diffie_hellman(&remote_public);
        let session = derive_session(
            SessionRole::Server,
            client.session_id,
            shared.as_bytes(),
            client,
            &hello,
        )?;

        Ok((hello, session))
    }
}

pub struct SecureSession {
    session_id: [u8; 16],
    tx_key: [u8; 32],
    rx_key: [u8; 32],
    tx_sequence: u64,
    rx_sequence: u64,
}

impl fmt::Debug for SecureSession {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SecureSession")
            .field("session_id", &self.session_id)
            .field("tx_sequence", &self.tx_sequence)
            .field("rx_sequence", &self.rx_sequence)
            .field("keys", &"<redacted>")
            .finish()
    }
}

impl SecureSession {
    pub fn session_id(&self) -> [u8; 16] {
        self.session_id
    }

    pub fn tx_sequence(&self) -> u64 {
        self.tx_sequence
    }

    pub fn rx_sequence(&self) -> u64 {
        self.rx_sequence
    }

    pub fn seal(&mut self, plaintext: &[u8]) -> Result<Vec<u8>, PcFabricError> {
        let sequence = self.tx_sequence;
        let next_sequence = sequence.checked_add(1).ok_or(PcFabricError::Overflow)?;
        let nonce = session_nonce(sequence);
        let mut header = Vec::with_capacity(SESSION_HEADER_LEN);
        header.extend_from_slice(&SESSION_FRAME_MAGIC);
        header.extend_from_slice(&SESSION_FRAME_MAJOR.to_le_bytes());
        header.extend_from_slice(&SESSION_FRAME_MINOR.to_le_bytes());
        header.extend_from_slice(&self.session_id);
        header.extend_from_slice(&sequence.to_le_bytes());
        header.extend_from_slice(&0u32.to_le_bytes());

        let cipher = ChaCha20Poly1305::new((&self.tx_key).into());
        let ciphertext = cipher
            .encrypt(
                Nonce::from_slice(&nonce),
                Payload {
                    msg: plaintext,
                    aad: &header[..SESSION_HEADER_LEN - 4],
                },
            )
            .map_err(|_| PcFabricError::CryptoFailure)?;
        let ciphertext_len =
            u32::try_from(ciphertext.len()).map_err(|_| PcFabricError::Overflow)?;
        header[SESSION_HEADER_LEN - 4..].copy_from_slice(&ciphertext_len.to_le_bytes());
        header.extend_from_slice(&ciphertext);
        self.tx_sequence = next_sequence;
        Ok(header)
    }

    pub fn open(&mut self, frame: &[u8]) -> Result<Vec<u8>, PcFabricError> {
        if frame.len() < SESSION_HEADER_LEN || frame[..6] != SESSION_FRAME_MAGIC {
            return Err(PcFabricError::InvalidFrame);
        }
        let major = u16::from_le_bytes([frame[6], frame[7]]);
        let minor = u16::from_le_bytes([frame[8], frame[9]]);
        if major != SESSION_FRAME_MAJOR || minor > SESSION_FRAME_MINOR {
            return Err(PcFabricError::InvalidFrame);
        }

        let mut session_id = [0u8; 16];
        session_id.copy_from_slice(&frame[10..26]);
        if session_id != self.session_id {
            return Err(PcFabricError::PeerMismatch);
        }

        let sequence = u64::from_le_bytes(
            frame[26..34]
                .try_into()
                .map_err(|_| PcFabricError::InvalidFrame)?,
        );
        if sequence != self.rx_sequence {
            return Err(PcFabricError::Replay {
                expected_sequence: self.rx_sequence,
                actual_sequence: sequence,
            });
        }

        let ciphertext_len = usize::try_from(u32::from_le_bytes(
            frame[34..38]
                .try_into()
                .map_err(|_| PcFabricError::InvalidFrame)?,
        ))
        .map_err(|_| PcFabricError::Overflow)?;
        let expected = SESSION_HEADER_LEN
            .checked_add(ciphertext_len)
            .ok_or(PcFabricError::Overflow)?;
        if frame.len() != expected {
            return Err(PcFabricError::InvalidFrame);
        }

        let nonce = session_nonce(sequence);
        let cipher = ChaCha20Poly1305::new((&self.rx_key).into());
        let plaintext = cipher
            .decrypt(
                Nonce::from_slice(&nonce),
                Payload {
                    msg: &frame[SESSION_HEADER_LEN..],
                    aad: &frame[..SESSION_HEADER_LEN - 4],
                },
            )
            .map_err(|_| PcFabricError::CryptoFailure)?;
        self.rx_sequence = self
            .rx_sequence
            .checked_add(1)
            .ok_or(PcFabricError::Overflow)?;
        Ok(plaintext)
    }
}

fn validate_identity_pair(
    local: &PairedIdentity,
    remote: &PairingRecord,
) -> Result<(), PcFabricError> {
    if peer_id_for_verify_key(&local.verify_key) != local.peer_id {
        return Err(PcFabricError::InvalidIdentity);
    }
    if peer_id_for_verify_key(&remote.remote_verify_key) != remote.remote_peer_id {
        return Err(PcFabricError::InvalidPairing);
    }
    if local.peer_id == remote.remote_peer_id {
        return Err(PcFabricError::InvalidPairing);
    }
    Ok(())
}

fn derive_session(
    role: SessionRole,
    session_id: [u8; 16],
    shared_secret: &[u8; 32],
    client: &ClientHello,
    server: &ServerHello,
) -> Result<SecureSession, PcFabricError> {
    let mut transcript = client_wire_bytes(client);
    transcript.extend_from_slice(&server_wire_bytes(server));
    let salt = digest(&transcript);
    let hkdf = Hkdf::<Sha256>::new(Some(&salt), shared_secret);
    let mut material = [0u8; 64];
    hkdf.expand(SESSION_KDF_INFO, &mut material)
        .map_err(|_| PcFabricError::CryptoFailure)?;

    let mut first = [0u8; 32];
    first.copy_from_slice(&material[..32]);
    let mut second = [0u8; 32];
    second.copy_from_slice(&material[32..]);

    let (tx_key, rx_key) = match role {
        SessionRole::Client => (first, second),
        SessionRole::Server => (second, first),
    };

    Ok(SecureSession {
        session_id,
        tx_key,
        rx_key,
        tx_sequence: 0,
        rx_sequence: 0,
    })
}

fn peer_id_for_verify_key(verify_key: &[u8; 32]) -> [u8; 16] {
    let mut hasher = Sha256::new();
    hasher.update(b"NTD97-PCF97-PEER");
    hasher.update(verify_key);
    let digest = hasher.finalize();
    let mut id = [0u8; 16];
    id.copy_from_slice(&digest[..16]);
    id
}

fn session_id(
    client_peer_id: &[u8; 16],
    server_peer_id: &[u8; 16],
    ephemeral_public: &[u8; 32],
    challenge: &[u8; 32],
) -> [u8; 16] {
    let mut hasher = Sha256::new();
    hasher.update(b"NTD97-PCF97-SESSION-ID");
    hasher.update(client_peer_id);
    hasher.update(server_peer_id);
    hasher.update(ephemeral_public);
    hasher.update(challenge);
    let digest = hasher.finalize();
    let mut id = [0u8; 16];
    id.copy_from_slice(&digest[..16]);
    id
}

fn derive32(label: &[u8], seed: &[u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(HANDSHAKE_DOMAIN);
    hasher.update(label);
    hasher.update(seed);
    hasher.finalize().into()
}

fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn client_signing_bytes(hello: &ClientHello) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(HANDSHAKE_DOMAIN);
    out.extend_from_slice(b"client");
    out.extend_from_slice(&hello.client_peer_id);
    out.extend_from_slice(&hello.expected_server_peer_id);
    out.extend_from_slice(&hello.session_id);
    out.extend_from_slice(&hello.ephemeral_public);
    out.extend_from_slice(&hello.challenge);
    out
}

fn server_signing_bytes(hello: &ServerHello) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(HANDSHAKE_DOMAIN);
    out.extend_from_slice(b"server");
    out.extend_from_slice(&hello.server_peer_id);
    out.extend_from_slice(&hello.expected_client_peer_id);
    out.extend_from_slice(&hello.session_id);
    out.extend_from_slice(&hello.ephemeral_public);
    out.extend_from_slice(&hello.challenge);
    out.extend_from_slice(&hello.client_hello_digest);
    out
}

fn client_wire_bytes(hello: &ClientHello) -> Vec<u8> {
    let mut out = client_signing_bytes(hello);
    out.extend_from_slice(&hello.signature);
    out
}

fn server_wire_bytes(hello: &ServerHello) -> Vec<u8> {
    let mut out = server_signing_bytes(hello);
    out.extend_from_slice(&hello.signature);
    out
}

fn session_nonce(sequence: u64) -> [u8; 12] {
    let mut nonce = [0u8; 12];
    nonce[..4].copy_from_slice(b"N97S");
    nonce[4..].copy_from_slice(&sequence.to_le_bytes());
    nonce
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sessions() -> (SecureSession, SecureSession) {
        let phone = PairedIdentity::from_seed([1; 32]);
        let pc = PairedIdentity::from_seed([2; 32]);
        let (client_state, client_hello) = ClientHandshake::begin(
            phone.clone(),
            pc.pairing_record(),
            HandshakeEntropy::from_seed([3; 32]),
        )
        .expect("client hello");
        let (server_hello, server_session) = ServerHello::accept(
            pc,
            phone.pairing_record(),
            HandshakeEntropy::from_seed([4; 32]),
            &client_hello,
        )
        .expect("server hello");
        let client_session = client_state.finish(&server_hello).expect("finish");
        (client_session, server_session)
    }

    #[test]
    fn mutual_authentication_derives_bidirectional_session() {
        let (mut client, mut server) = sessions();

        let frame = client.seal(b"typed remote task").expect("seal");
        assert_eq!(server.open(&frame).expect("open"), b"typed remote task");

        let reply = server.seal(b"verified result").expect("seal");
        assert_eq!(client.open(&reply).expect("open"), b"verified result");
    }

    #[test]
    fn replayed_frame_is_rejected() {
        let (mut client, mut server) = sessions();
        let frame = client.seal(b"once").expect("seal");
        assert_eq!(server.open(&frame).expect("first"), b"once");
        assert!(matches!(
            server.open(&frame),
            Err(PcFabricError::Replay {
                expected_sequence: 1,
                actual_sequence: 0
            })
        ));
    }

    #[test]
    fn hello_for_unpaired_identity_is_rejected() {
        let phone = PairedIdentity::from_seed([1; 32]);
        let pc = PairedIdentity::from_seed([2; 32]);
        let attacker = PairedIdentity::from_seed([9; 32]);

        let (_, hello) = ClientHandshake::begin(
            attacker,
            pc.pairing_record(),
            HandshakeEntropy::from_seed([3; 32]),
        )
        .expect("attacker hello");

        assert!(matches!(
            ServerHello::accept(
                pc,
                phone.pairing_record(),
                HandshakeEntropy::from_seed([4; 32]),
                &hello
            ),
            Err(PcFabricError::PeerMismatch)
        ));
    }
}
