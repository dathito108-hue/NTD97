#![forbid(unsafe_code)]

mod adapter;
mod agent;
mod artifact;
mod crypto;
mod transport;
mod wire;

pub use adapter::{PairedPcAdapter, PairedPcTransport};
pub use agent::{
    DesktopAgent, DesktopArtifactPolicy, DesktopCapabilityHandler, DesktopExecutionPolicy,
    DesktopHandlerOutput, FileArtifactHandler, ProcessExecutionHandler, SystemObserveHandler,
};
pub use artifact::{
    ArtifactAssembler, ArtifactChunk, ArtifactDescriptor, ArtifactSender, DEFAULT_ARTIFACT_CHUNK,
};
pub use crypto::{
    ClientHandshake, ClientHello, HandshakeEntropy, PairedIdentity, PairingRecord, SecureSession,
    ServerHello, SessionRole,
};
pub use transport::{
    read_length_prefixed_frame, write_length_prefixed_frame, TcpFrameTransport,
    DEFAULT_MAX_TRANSPORT_FRAME,
};
pub use wire::{
    decode_message, encode_message, RemoteAction, RemoteCapability, RemoteMessage, RemoteRequest,
    RemoteResult, RemoteResultStatus, PCF97_MAGIC, PCF97_MAJOR, PCF97_MINOR,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PcFabricError {
    InvalidIdentity,
    InvalidPairing,
    InvalidSignature,
    InvalidHandshake,
    CryptoFailure,
    Replay {
        expected_sequence: u64,
        actual_sequence: u64,
    },
    InvalidFrame,
    InvalidMessage,
    InvalidArtifact,
    ArtifactHashMismatch,
    ArtifactOutOfOrder,
    MissingCapability(String),
    CapabilityVersionMismatch {
        capability: String,
        expected: u32,
        actual: u32,
    },
    PeerMismatch,
    PolicyDenied(String),
    Io(String),
    Remote(String),
    Overflow,
}

impl From<std::io::Error> for PcFabricError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value.to_string())
    }
}
