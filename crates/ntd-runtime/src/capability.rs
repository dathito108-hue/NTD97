#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use ntd_core::{CapabilityId, SideEffectClass};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum CapabilityDomain {
    Web = 1,
    Browser = 2,
    File = 3,
    Device = 4,
    App = 5,
    Custom = 6,
}

impl TryFrom<u8> for CapabilityDomain {
    type Error = CapabilityError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Web),
            2 => Ok(Self::Browser),
            3 => Ok(Self::File),
            4 => Ok(Self::Device),
            5 => Ok(Self::App),
            6 => Ok(Self::Custom),
            other => Err(CapabilityError::InvalidDomain(other)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AuthorityScope(pub String);

impl AuthorityScope {
    pub fn new(value: impl Into<String>) -> Result<Self, CapabilityError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(CapabilityError::EmptyScope);
        }
        Ok(Self(value))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthorityGrant {
    scopes: BTreeSet<AuthorityScope>,
    pub allow_external_write: bool,
    pub allow_irreversible: bool,
}

impl Default for AuthorityGrant {
    fn default() -> Self {
        Self::new()
    }
}

impl AuthorityGrant {
    pub fn new() -> Self {
        Self {
            scopes: BTreeSet::new(),
            allow_external_write: false,
            allow_irreversible: false,
        }
    }

    pub fn with_scope(mut self, scope: AuthorityScope) -> Self {
        self.scopes.insert(scope);
        self
    }

    pub fn scopes(&self) -> &BTreeSet<AuthorityScope> {
        &self.scopes
    }

    pub fn permits(&self, descriptor: &CapabilityDescriptor) -> Result<(), CapabilityError> {
        for required in &descriptor.required_scopes {
            if !self.scopes.contains(required) {
                return Err(CapabilityError::MissingAuthority(required.clone()));
            }
        }

        match descriptor.side_effect {
            SideEffectClass::ReadOnly | SideEffectClass::Reversible => Ok(()),
            SideEffectClass::ExternalWrite if self.allow_external_write => Ok(()),
            SideEffectClass::ExternalWrite => Err(CapabilityError::ExternalWriteNotAuthorized),
            SideEffectClass::Irreversible if self.allow_irreversible => Ok(()),
            SideEffectClass::Irreversible => Err(CapabilityError::IrreversibleNotAuthorized),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityDescriptor {
    pub id: CapabilityId,
    pub version: u32,
    pub domain: CapabilityDomain,
    pub side_effect: SideEffectClass,
    pub verification_required: bool,
    pub rollback_supported: bool,
    pub resumable: bool,
    pub required_scopes: Vec<AuthorityScope>,
}

impl CapabilityDescriptor {
    pub fn new(
        id: CapabilityId,
        version: u32,
        domain: CapabilityDomain,
        side_effect: SideEffectClass,
    ) -> Result<Self, CapabilityError> {
        if id.0.trim().is_empty() {
            return Err(CapabilityError::EmptyCapabilityId);
        }
        if version == 0 {
            return Err(CapabilityError::InvalidVersion);
        }

        Ok(Self {
            id,
            version,
            domain,
            side_effect,
            verification_required: true,
            rollback_supported: false,
            resumable: false,
            required_scopes: Vec::new(),
        })
    }

    pub fn normalize(&mut self) -> Result<(), CapabilityError> {
        if self.id.0.trim().is_empty() {
            return Err(CapabilityError::EmptyCapabilityId);
        }
        if self.version == 0 {
            return Err(CapabilityError::InvalidVersion);
        }

        self.required_scopes.sort();
        self.required_scopes.dedup();

        if self.side_effect == SideEffectClass::Irreversible && self.rollback_supported {
            return Err(CapabilityError::IrreversibleRollback);
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypedAction {
    WebSearch {
        query: String,
        max_results: u16,
    },
    WebFetch {
        url: String,
    },
    BrowserObserve {
        target: String,
    },
    BrowserInteract {
        target: String,
        operation: String,
        value: Option<String>,
    },
    FileRead {
        path: String,
    },
    FileWrite {
        path: String,
        bytes: Vec<u8>,
    },
    DeviceObserve {
        surface: String,
    },
    DeviceInteract {
        surface: String,
        operation: String,
        argument: Option<String>,
    },
    AppAction {
        app: String,
        action: String,
        payload: Vec<u8>,
    },
    Custom {
        type_name: String,
        payload: Vec<u8>,
    },
}

impl TypedAction {
    pub fn domain(&self) -> CapabilityDomain {
        match self {
            Self::WebSearch { .. } | Self::WebFetch { .. } => CapabilityDomain::Web,
            Self::BrowserObserve { .. } | Self::BrowserInteract { .. } => CapabilityDomain::Browser,
            Self::FileRead { .. } | Self::FileWrite { .. } => CapabilityDomain::File,
            Self::DeviceObserve { .. } | Self::DeviceInteract { .. } => CapabilityDomain::Device,
            Self::AppAction { .. } => CapabilityDomain::App,
            Self::Custom { .. } => CapabilityDomain::Custom,
        }
    }

    pub fn validate(&self) -> Result<(), CapabilityError> {
        match self {
            Self::WebSearch { query, max_results } => {
                nonempty(query)?;
                if *max_results == 0 {
                    return Err(CapabilityError::InvalidAction);
                }
            }
            Self::WebFetch { url } => nonempty(url)?,
            Self::BrowserObserve { target } => nonempty(target)?,
            Self::BrowserInteract {
                target,
                operation,
                ..
            } => {
                nonempty(target)?;
                nonempty(operation)?;
            }
            Self::FileRead { path } => nonempty(path)?,
            Self::FileWrite { path, .. } => nonempty(path)?,
            Self::DeviceObserve { surface } => nonempty(surface)?,
            Self::DeviceInteract {
                surface,
                operation,
                ..
            } => {
                nonempty(surface)?;
                nonempty(operation)?;
            }
            Self::AppAction { app, action, .. } => {
                nonempty(app)?;
                nonempty(action)?;
            }
            Self::Custom { type_name, .. } => nonempty(type_name)?,
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionValue {
    None,
    Text(String),
    Bytes(Vec<u8>),
    TextList(Vec<String>),
    Fields(BTreeMap<String, String>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionOutput {
    pub summary: String,
    pub value: ActionValue,
    pub evidence: Vec<String>,
}

impl ActionOutput {
    pub fn text(summary: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            summary: summary.into(),
            value: ActionValue::Text(value.into()),
            evidence: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapabilityError {
    EmptyCapabilityId,
    InvalidVersion,
    EmptyScope,
    InvalidDomain(u8),
    InvalidAction,
    DomainMismatch {
        capability: CapabilityId,
        expected: CapabilityDomain,
        actual: CapabilityDomain,
    },
    DuplicateCapability(CapabilityId),
    MissingCapability(CapabilityId),
    MissingAuthority(AuthorityScope),
    ExternalWriteNotAuthorized,
    IrreversibleNotAuthorized,
    IrreversibleRollback,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CapabilityRegistry {
    descriptors: BTreeMap<String, CapabilityDescriptor>,
}

impl CapabilityRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(
        &mut self,
        mut descriptor: CapabilityDescriptor,
    ) -> Result<(), CapabilityError> {
        descriptor.normalize()?;
        let key = descriptor.id.0.clone();
        if self.descriptors.contains_key(&key) {
            return Err(CapabilityError::DuplicateCapability(descriptor.id));
        }
        self.descriptors.insert(key, descriptor);
        Ok(())
    }

    pub fn descriptor(
        &self,
        id: &CapabilityId,
    ) -> Result<&CapabilityDescriptor, CapabilityError> {
        self.descriptors
            .get(&id.0)
            .ok_or_else(|| CapabilityError::MissingCapability(id.clone()))
    }

    pub fn validate_action(
        &self,
        id: &CapabilityId,
        action: &TypedAction,
    ) -> Result<&CapabilityDescriptor, CapabilityError> {
        action.validate()?;
        let descriptor = self.descriptor(id)?;
        let actual = action.domain();
        if descriptor.domain != actual {
            return Err(CapabilityError::DomainMismatch {
                capability: id.clone(),
                expected: descriptor.domain,
                actual,
            });
        }
        Ok(descriptor)
    }

    pub fn descriptors(&self) -> impl Iterator<Item = &CapabilityDescriptor> {
        self.descriptors.values()
    }
}

fn nonempty(value: &str) -> Result<(), CapabilityError> {
    if value.trim().is_empty() {
        Err(CapabilityError::InvalidAction)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_descriptor() -> CapabilityDescriptor {
        let mut descriptor = CapabilityDescriptor::new(
            CapabilityId("web.search".into()),
            1,
            CapabilityDomain::Web,
            SideEffectClass::ReadOnly,
        )
        .expect("descriptor");
        descriptor
            .required_scopes
            .push(AuthorityScope::new("network.read").expect("scope"));
        descriptor
    }

    #[test]
    fn authority_requires_declared_scope() {
        let descriptor = read_descriptor();
        assert!(matches!(
            AuthorityGrant::new().permits(&descriptor),
            Err(CapabilityError::MissingAuthority(_))
        ));

        let grant =
            AuthorityGrant::new().with_scope(AuthorityScope::new("network.read").expect("scope"));
        assert_eq!(grant.permits(&descriptor), Ok(()));
    }

    #[test]
    fn irreversible_requires_explicit_authority() {
        let descriptor = CapabilityDescriptor::new(
            CapabilityId("device.factory-reset".into()),
            1,
            CapabilityDomain::Device,
            SideEffectClass::Irreversible,
        )
        .expect("descriptor");

        assert_eq!(
            AuthorityGrant::new().permits(&descriptor),
            Err(CapabilityError::IrreversibleNotAuthorized)
        );

        let mut grant = AuthorityGrant::new();
        grant.allow_irreversible = true;
        assert_eq!(grant.permits(&descriptor), Ok(()));
    }

    #[test]
    fn registry_rejects_domain_mismatch() {
        let mut registry = CapabilityRegistry::new();
        registry.register(read_descriptor()).expect("register");

        assert!(matches!(
            registry.validate_action(
                &CapabilityId("web.search".into()),
                &TypedAction::FileRead {
                    path: "/tmp/x".into()
                }
            ),
            Err(CapabilityError::DomainMismatch { .. })
        ));
    }
}
