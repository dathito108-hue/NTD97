#![forbid(unsafe_code)]

use ntd_runtime::{
    ActionId, ActionOutput, ActionValue, AdapterResult, CapabilityAdapter, TypedAction,
};

use crate::{
    decode_message, encode_message, ArtifactAssembler, ArtifactDescriptor, PcFabricError,
    RemoteAction, RemoteCapability, RemoteMessage, RemoteRequest, RemoteResultStatus,
    SecureSession,
};

pub trait PairedPcTransport {
    fn exchange(&mut self, encrypted_frame: &[u8]) -> Result<Vec<u8>, PcFabricError>;
}

pub struct PairedPcAdapter<T> {
    peer: String,
    capability: String,
    capability_version: u32,
    session: SecureSession,
    transport: T,
}

impl<T> PairedPcAdapter<T>
where
    T: PairedPcTransport,
{
    pub fn new(
        peer: impl Into<String>,
        capability: impl Into<String>,
        capability_version: u32,
        session: SecureSession,
        transport: T,
    ) -> Result<Self, PcFabricError> {
        let peer = peer.into();
        let capability = capability.into();
        if peer.trim().is_empty() || capability.trim().is_empty() || capability_version == 0 {
            return Err(PcFabricError::InvalidMessage);
        }
        Ok(Self {
            peer,
            capability,
            capability_version,
            session,
            transport,
        })
    }

    pub fn discover_capabilities(&mut self) -> Result<Vec<RemoteCapability>, PcFabricError> {
        let response = self.exchange_message(RemoteMessage::CapabilityQuery)?;
        let RemoteMessage::Capabilities(capabilities) = response else {
            return Err(PcFabricError::InvalidMessage);
        };
        Ok(capabilities)
    }

    pub fn peer(&self) -> &str {
        &self.peer
    }

    pub fn capability(&self) -> &str {
        &self.capability
    }

    fn exchange_message(&mut self, message: RemoteMessage) -> Result<RemoteMessage, PcFabricError> {
        let plaintext = encode_message(&message)?;
        let outbound = self.session.seal(&plaintext)?;
        let inbound = self.transport.exchange(&outbound)?;
        let plaintext = self.session.open(&inbound)?;
        decode_message(&plaintext)
    }

    fn map_action(&self, action: &TypedAction) -> Result<RemoteAction, PcFabricError> {
        match action {
            TypedAction::PcObserve { peer, surface } if peer == &self.peer => {
                Ok(RemoteAction::Observe {
                    surface: surface.clone(),
                })
            }
            TypedAction::PcExecute {
                peer,
                program,
                args,
                working_dir,
            } if peer == &self.peer => Ok(RemoteAction::Execute {
                program: program.clone(),
                args: args.clone(),
                working_dir: working_dir.clone(),
            }),
            TypedAction::PcArtifactRead { peer, path } if peer == &self.peer => {
                Ok(RemoteAction::ArtifactRead { path: path.clone() })
            }
            TypedAction::PcArtifactWrite { peer, path, bytes } if peer == &self.peer => {
                Ok(RemoteAction::ArtifactWrite {
                    path: path.clone(),
                    bytes: bytes.clone(),
                })
            }
            TypedAction::PcObserve { .. }
            | TypedAction::PcExecute { .. }
            | TypedAction::PcArtifactRead { .. }
            | TypedAction::PcArtifactWrite { .. } => Err(PcFabricError::PeerMismatch),
            _ => Err(PcFabricError::InvalidMessage),
        }
    }

    fn pull_artifact(&mut self, descriptor: ArtifactDescriptor) -> Result<Vec<u8>, PcFabricError> {
        let mut assembler = ArtifactAssembler::new(descriptor.clone())?;
        while assembler.next_offset() < descriptor.length {
            let response = self.exchange_message(RemoteMessage::ArtifactPull {
                transfer_id: descriptor.transfer_id,
                offset: assembler.next_offset(),
            })?;
            let RemoteMessage::ArtifactChunk(chunk) = response else {
                return Err(PcFabricError::InvalidMessage);
            };
            assembler.push(chunk)?;
        }
        assembler.finish()
    }

    fn materialize_artifact_output(
        &mut self,
        action: &TypedAction,
        mut output: ActionOutput,
        artifacts: Vec<ArtifactDescriptor>,
    ) -> Result<ActionOutput, PcFabricError> {
        if matches!(action, TypedAction::PcArtifactRead { .. }) {
            if artifacts.len() != 1 {
                return Err(PcFabricError::InvalidArtifact);
            }
            let descriptor = artifacts[0].clone();
            let bytes = self.pull_artifact(descriptor.clone())?;
            output.value = ActionValue::Bytes(bytes);
            output
                .evidence
                .push(format!("sha256={:02x?}", descriptor.sha256));
        } else if !artifacts.is_empty() {
            for artifact in artifacts {
                output.evidence.push(format!(
                    "artifact:{}:{}",
                    artifact.transfer_id, artifact.name
                ));
            }
        }
        Ok(output)
    }
}

impl<T> CapabilityAdapter for PairedPcAdapter<T>
where
    T: PairedPcTransport + 'static,
{
    fn execute(
        &mut self,
        action_id: ActionId,
        action: &TypedAction,
    ) -> Result<AdapterResult, String> {
        let remote_action = self
            .map_action(action)
            .map_err(|error| format!("{error:?}"))?;
        let request = RemoteRequest::new(
            action_id.0,
            self.capability.clone(),
            self.capability_version,
            remote_action,
        )
        .map_err(|error| format!("{error:?}"))?;

        let response = self
            .exchange_message(RemoteMessage::Request(request.clone()))
            .map_err(|error| format!("{error:?}"))?;
        let RemoteMessage::Result(result) = response else {
            return Err("paired PC returned an unexpected message".into());
        };
        result
            .verify_against(&request)
            .map_err(|error| format!("{error:?}"))?;

        match result.status {
            RemoteResultStatus::Completed => {
                let output = self
                    .materialize_artifact_output(action, result.output, result.artifacts)
                    .map_err(|error| format!("{error:?}"))?;
                Ok(AdapterResult::Completed {
                    output,
                    rollback_token: None,
                })
            }
            RemoteResultStatus::Retryable(reason) => Ok(AdapterResult::Retryable {
                reason,
                resume_token: None,
            }),
            RemoteResultStatus::Rejected(reason) => Err(reason),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use ntd_core::SideEffectClass;
    use ntd_runtime::{ActionOutput, ActionValue};

    use crate::{
        ClientHandshake, DesktopAgent, DesktopCapabilityHandler, DesktopHandlerOutput,
        HandshakeEntropy, PairedIdentity, RemoteCapability, ServerHello, SystemObserveHandler,
    };

    use super::*;

    struct LoopbackTransport {
        agent: Rc<RefCell<DesktopAgent>>,
    }

    impl PairedPcTransport for LoopbackTransport {
        fn exchange(&mut self, encrypted_frame: &[u8]) -> Result<Vec<u8>, PcFabricError> {
            self.agent
                .borrow_mut()
                .handle_encrypted_frame(encrypted_frame)
        }
    }

    struct CountingHandler {
        calls: Rc<RefCell<u32>>,
    }

    impl DesktopCapabilityHandler for CountingHandler {
        fn capability(&self) -> RemoteCapability {
            RemoteCapability {
                id: "pc.test.execute".into(),
                version: 1,
                side_effect: SideEffectClass::ExternalWrite,
                verification_required: true,
                rollback_supported: false,
                resumable: false,
                required_scopes: vec!["pc.execute".into()],
            }
        }

        fn execute(
            &mut self,
            action: &RemoteAction,
        ) -> Result<DesktopHandlerOutput, PcFabricError> {
            if !matches!(action, RemoteAction::Execute { .. }) {
                return Err(PcFabricError::InvalidMessage);
            }
            *self.calls.borrow_mut() += 1;
            Ok(DesktopHandlerOutput::new(ActionOutput {
                summary: "executed".into(),
                value: ActionValue::Text("ok".into()),
                evidence: Vec::new(),
            }))
        }
    }

    fn paired_agent() -> (SecureSession, Rc<RefCell<DesktopAgent>>) {
        let phone = PairedIdentity::from_seed([31; 32]);
        let pc = PairedIdentity::from_seed([32; 32]);
        let (client_state, client_hello) = ClientHandshake::begin(
            phone.clone(),
            pc.pairing_record(),
            HandshakeEntropy::from_seed([33; 32]),
        )
        .expect("begin");
        let (server_hello, server_session) = ServerHello::accept(
            pc,
            phone.pairing_record(),
            HandshakeEntropy::from_seed([34; 32]),
            &client_hello,
        )
        .expect("accept");
        let client_session = client_state.finish(&server_hello).expect("finish");
        (
            client_session,
            Rc::new(RefCell::new(DesktopAgent::new(server_session))),
        )
    }

    #[test]
    fn adapter_discovers_and_executes_typed_remote_capability() {
        let (session, agent) = paired_agent();
        agent
            .borrow_mut()
            .register_handler(SystemObserveHandler)
            .expect("handler");
        let transport = LoopbackTransport {
            agent: Rc::clone(&agent),
        };
        let mut adapter =
            PairedPcAdapter::new("workstation", "pc.system.observe", 1, session, transport)
                .expect("adapter");

        let capabilities = adapter.discover_capabilities().expect("discover");
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
    }

    #[test]
    fn stable_action_id_does_not_replay_remote_side_effect() {
        let (session, agent) = paired_agent();
        let calls = Rc::new(RefCell::new(0));
        agent
            .borrow_mut()
            .register_handler(CountingHandler {
                calls: Rc::clone(&calls),
            })
            .expect("handler");

        let transport = LoopbackTransport {
            agent: Rc::clone(&agent),
        };
        let mut adapter =
            PairedPcAdapter::new("workstation", "pc.test.execute", 1, session, transport)
                .expect("adapter");
        let action = TypedAction::PcExecute {
            peer: "workstation".into(),
            program: "unit-test".into(),
            args: Vec::new(),
            working_dir: None,
        };

        adapter.execute(ActionId(41), &action).expect("first");
        adapter.execute(ActionId(41), &action).expect("retry");

        assert_eq!(*calls.borrow(), 1);
    }
}
