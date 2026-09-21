#![forbid(unsafe_code)]

use std::{
    io::{Read, Write},
    net::{TcpStream, ToSocketAddrs},
};

use crate::{adapter::PairedPcTransport, PcFabricError};

pub const DEFAULT_MAX_TRANSPORT_FRAME: usize = 16 * 1024 * 1024;

pub fn write_length_prefixed_frame<W: Write>(
    writer: &mut W,
    frame: &[u8],
) -> Result<(), PcFabricError> {
    if frame.is_empty() || frame.len() > DEFAULT_MAX_TRANSPORT_FRAME {
        return Err(PcFabricError::InvalidFrame);
    }
    let len = u32::try_from(frame.len()).map_err(|_| PcFabricError::Overflow)?;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(frame)?;
    writer.flush()?;
    Ok(())
}

pub fn read_length_prefixed_frame<R: Read>(reader: &mut R) -> Result<Vec<u8>, PcFabricError> {
    let mut len_bytes = [0u8; 4];
    reader.read_exact(&mut len_bytes)?;
    let len = usize::try_from(u32::from_le_bytes(len_bytes)).map_err(|_| PcFabricError::Overflow)?;
    if len == 0 || len > DEFAULT_MAX_TRANSPORT_FRAME {
        return Err(PcFabricError::InvalidFrame);
    }
    let mut frame = vec![0u8; len];
    reader.read_exact(&mut frame)?;
    Ok(frame)
}

#[derive(Debug)]
pub struct TcpFrameTransport {
    stream: TcpStream,
}

impl TcpFrameTransport {
    pub fn connect(address: impl ToSocketAddrs) -> Result<Self, PcFabricError> {
        let stream = TcpStream::connect(address)?;
        stream.set_nodelay(true)?;
        Ok(Self { stream })
    }

    pub fn from_stream(stream: TcpStream) -> Result<Self, PcFabricError> {
        stream.set_nodelay(true)?;
        Ok(Self { stream })
    }

    pub fn stream(&self) -> &TcpStream {
        &self.stream
    }
}

impl PairedPcTransport for TcpFrameTransport {
    fn exchange(&mut self, encrypted_frame: &[u8]) -> Result<Vec<u8>, PcFabricError> {
        write_length_prefixed_frame(&mut self.stream, encrypted_frame)?;
        read_length_prefixed_frame(&mut self.stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_prefixed_frame_round_trips() {
        let frame = b"encrypted-pcf97-frame";
        let mut bytes = Vec::new();
        write_length_prefixed_frame(&mut bytes, frame).expect("write");
        let decoded = read_length_prefixed_frame(&mut bytes.as_slice()).expect("read");
        assert_eq!(decoded, frame);
    }
}
