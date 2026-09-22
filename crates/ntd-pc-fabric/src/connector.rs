#![forbid(unsafe_code)]

use std::{
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    time::Duration,
};

use crate::{
    ClientHandshake, ClientHello, HandshakeEntropy, PairedIdentity, PairingRecord, PcFabricError,
    SecureSession, ServerHello, TcpFrameTransport,
};

const HANDSHAKE_MAGIC: [u8; 6] = *b"PCH97\0";
const HANDSHAKE_MAJOR: u16 = 0;
const HANDSHAKE_MINOR: u16 = 1;
const HANDSHAKE_HEADER_LEN: usize = 16;
const CLIENT_HELLO_TAG: u8 = 1;
const SERVER_HELLO_TAG: u8 = 2;
const CLIENT_HELLO_LEN: usize = 16 + 16 + 16 + 32 + 32 + 64;
const SERVER_HELLO_LEN: usize = 16 + 16 + 16 + 32 + 32 + 32 + 64;

pub fn connect_paired_tcp(
    address: SocketAddr,
    local: PairedIdentity,
    remote: PairingRecord,
    entropy: HandshakeEntropy,
    timeout: Duration,
) -> Result<(SecureSession, TcpFrameTransport), PcFabricError> {
    if timeout.is_zero() {
        return Err(PcFabricError::InvalidHandshake);
    }
    let mut stream = TcpStream::connect_timeout(&address, timeout)?;
    configure_stream(&stream, timeout)?;

    let (state, hello) = ClientHandshake::begin(local, remote, entropy)?;
    let encoded = encode_client_hello(&hello)?;
    write_handshake_frame(&mut stream, &encoded)?;
    let server = decode_server_hello(&read_handshake_frame(&mut stream)?)?;
    let session = state.finish(&server)?;
    let transport = TcpFrameTransport::from_stream(stream)?;
    Ok((session, transport))
}

pub fn accept_paired_tcp(
    stream: &mut TcpStream,
    local: PairedIdentity,
    remote: PairingRecord,
    entropy: HandshakeEntropy,
    timeout: Duration,
) -> Result<SecureSession, PcFabricError> {
    if timeout.is_zero() {
        return Err(PcFabricError::InvalidHandshake);
    }
    configure_stream(stream, timeout)?;
    let client = decode_client_hello(&read_handshake_frame(stream)?)?;
    let (server, session) = ServerHello::accept(local, remote, entropy, &client)?;
    let encoded = encode_server_hello(&server)?;
    write_handshake_frame(stream, &encoded)?;
    Ok(session)
}

pub fn encode_client_hello(hello: &ClientHello) -> Result<Vec<u8>, PcFabricError> {
    let mut payload = Vec::with_capacity(CLIENT_HELLO_LEN);
    payload.extend_from_slice(&hello.client_peer_id);
    payload.extend_from_slice(&hello.expected_server_peer_id);
    payload.extend_from_slice(&hello.session_id);
    payload.extend_from_slice(&hello.ephemeral_public);
    payload.extend_from_slice(&hello.challenge);
    payload.extend_from_slice(&hello.signature);
    encode_handshake(CLIENT_HELLO_TAG, &payload)
}

pub fn decode_client_hello(bytes: &[u8]) -> Result<ClientHello, PcFabricError> {
    let payload = decode_handshake(bytes, CLIENT_HELLO_TAG, CLIENT_HELLO_LEN)?;
    let mut cursor = FixedCursor::new(payload);
    Ok(ClientHello {
        client_peer_id: cursor.array()?,
        expected_server_peer_id: cursor.array()?,
        session_id: cursor.array()?,
        ephemeral_public: cursor.array()?,
        challenge: cursor.array()?,
        signature: cursor.array()?,
    })
}

pub fn encode_server_hello(hello: &ServerHello) -> Result<Vec<u8>, PcFabricError> {
    let mut payload = Vec::with_capacity(SERVER_HELLO_LEN);
    payload.extend_from_slice(&hello.server_peer_id);
    payload.extend_from_slice(&hello.expected_client_peer_id);
    payload.extend_from_slice(&hello.session_id);
    payload.extend_from_slice(&hello.ephemeral_public);
    payload.extend_from_slice(&hello.challenge);
    payload.extend_from_slice(&hello.client_hello_digest);
    payload.extend_from_slice(&hello.signature);
    encode_handshake(SERVER_HELLO_TAG, &payload)
}

pub fn decode_server_hello(bytes: &[u8]) -> Result<ServerHello, PcFabricError> {
    let payload = decode_handshake(bytes, SERVER_HELLO_TAG, SERVER_HELLO_LEN)?;
    let mut cursor = FixedCursor::new(payload);
    Ok(ServerHello {
        server_peer_id: cursor.array()?,
        expected_client_peer_id: cursor.array()?,
        session_id: cursor.array()?,
        ephemeral_public: cursor.array()?,
        challenge: cursor.array()?,
        client_hello_digest: cursor.array()?,
        signature: cursor.array()?,
    })
}

fn configure_stream(stream: &TcpStream, timeout: Duration) -> Result<(), PcFabricError> {
    stream.set_nodelay(true)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    Ok(())
}

fn encode_handshake(tag: u8, payload: &[u8]) -> Result<Vec<u8>, PcFabricError> {
    let len = u32::try_from(payload.len()).map_err(|_| PcFabricError::Overflow)?;
    let mut out = Vec::with_capacity(
        HANDSHAKE_HEADER_LEN
            .checked_add(payload.len())
            .ok_or(PcFabricError::Overflow)?,
    );
    out.extend_from_slice(&HANDSHAKE_MAGIC);
    out.extend_from_slice(&HANDSHAKE_MAJOR.to_le_bytes());
    out.extend_from_slice(&HANDSHAKE_MINOR.to_le_bytes());
    out.push(tag);
    out.push(0);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(payload);
    Ok(out)
}

fn decode_handshake(
    bytes: &[u8],
    expected_tag: u8,
    expected_payload_len: usize,
) -> Result<&[u8], PcFabricError> {
    if bytes.len() < HANDSHAKE_HEADER_LEN || bytes[..6] != HANDSHAKE_MAGIC {
        return Err(PcFabricError::InvalidHandshake);
    }
    let major = u16::from_le_bytes([bytes[6], bytes[7]]);
    let minor = u16::from_le_bytes([bytes[8], bytes[9]]);
    if major != HANDSHAKE_MAJOR
        || minor > HANDSHAKE_MINOR
        || bytes[10] != expected_tag
        || bytes[11] != 0
    {
        return Err(PcFabricError::InvalidHandshake);
    }
    let payload_len = usize::try_from(u32::from_le_bytes(
        bytes[12..16]
            .try_into()
            .map_err(|_| PcFabricError::InvalidHandshake)?,
    ))
    .map_err(|_| PcFabricError::Overflow)?;
    if payload_len != expected_payload_len || bytes.len() != HANDSHAKE_HEADER_LEN + payload_len {
        return Err(PcFabricError::InvalidHandshake);
    }
    Ok(&bytes[HANDSHAKE_HEADER_LEN..])
}

fn write_handshake_frame(stream: &mut TcpStream, bytes: &[u8]) -> Result<(), PcFabricError> {
    let len = u32::try_from(bytes.len()).map_err(|_| PcFabricError::Overflow)?;
    stream.write_all(&len.to_le_bytes())?;
    stream.write_all(bytes)?;
    stream.flush()?;
    Ok(())
}

fn read_handshake_frame(stream: &mut TcpStream) -> Result<Vec<u8>, PcFabricError> {
    let mut length = [0u8; 4];
    stream.read_exact(&mut length)?;
    let len = usize::try_from(u32::from_le_bytes(length)).map_err(|_| PcFabricError::Overflow)?;
    if !(HANDSHAKE_HEADER_LEN..=4096).contains(&len) {
        return Err(PcFabricError::InvalidHandshake);
    }
    let mut bytes = vec![0u8; len];
    stream.read_exact(&mut bytes)?;
    Ok(bytes)
}

struct FixedCursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> FixedCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], PcFabricError> {
        let end = self
            .position
            .checked_add(N)
            .ok_or(PcFabricError::Overflow)?;
        let slice = self
            .bytes
            .get(self.position..end)
            .ok_or(PcFabricError::InvalidHandshake)?;
        self.position = end;
        slice
            .try_into()
            .map_err(|_| PcFabricError::InvalidHandshake)
    }
}

#[cfg(test)]
mod tests {
    use std::{net::TcpListener, thread};

    use ntd_runtime::{ActionId, AdapterResult, TypedAction};

    use crate::{
        read_length_prefixed_frame, write_length_prefixed_frame, DesktopAgent, PairedPcAdapter,
        SystemObserveHandler,
    };

    use super::*;

    #[test]
    fn handshake_codec_round_trips() {
        let phone = PairedIdentity::from_seed([1; 32]);
        let pc = PairedIdentity::from_seed([2; 32]);
        let (_, client) = ClientHandshake::begin(
            phone.clone(),
            pc.pairing_record(),
            HandshakeEntropy::from_seed([3; 32]),
        )
        .expect("client");
        let (server, _) = ServerHello::accept(
            pc,
            phone.pairing_record(),
            HandshakeEntropy::from_seed([4; 32]),
            &client,
        )
        .expect("server");

        assert_eq!(
            decode_client_hello(&encode_client_hello(&client).expect("encode")).expect("decode"),
            client
        );
        assert_eq!(
            decode_server_hello(&encode_server_hello(&server).expect("encode")).expect("decode"),
            server
        );
    }

    #[test]
    fn tcp_connector_executes_authenticated_remote_action() {
        let phone = PairedIdentity::from_seed([11; 32]);
        let pc = PairedIdentity::from_seed([12; 32]);
        let listener = TcpListener::bind("127.0.0.1:0").expect("listen");
        let address = listener.local_addr().expect("address");
        let server_phone = phone.clone();
        let server_pc = pc.clone();

        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let session = accept_paired_tcp(
                &mut stream,
                server_pc,
                server_phone.pairing_record(),
                HandshakeEntropy::from_seed([14; 32]),
                Duration::from_secs(3),
            )
            .expect("server handshake");
            let mut agent = DesktopAgent::new(session);
            agent
                .register_handler(SystemObserveHandler)
                .expect("register observe");
            for _ in 0..2 {
                let frame = read_length_prefixed_frame(&mut stream).expect("read request");
                let reply = agent.handle_encrypted_frame(&frame).expect("agent");
                write_length_prefixed_frame(&mut stream, &reply).expect("write response");
            }
        });

        let (session, transport) = connect_paired_tcp(
            address,
            phone,
            pc.pairing_record(),
            HandshakeEntropy::from_seed([13; 32]),
            Duration::from_secs(3),
        )
        .expect("client handshake");
        let mut adapter =
            PairedPcAdapter::new("workstation", "pc.system.observe", 1, session, transport)
                .expect("adapter");
        let capabilities = adapter.discover_capabilities().expect("discover");
        assert_eq!(capabilities.len(), 1);
        assert_eq!(capabilities[0].id, "pc.system.observe");
        let result = adapter
            .execute(
                ActionId(7),
                &TypedAction::PcObserve {
                    peer: "workstation".into(),
                    surface: "system".into(),
                },
            )
            .expect("execute");
        assert!(matches!(result, AdapterResult::Completed { .. }));
        server.join().expect("server");
    }
}
