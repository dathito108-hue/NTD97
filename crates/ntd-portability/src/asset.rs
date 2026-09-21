#![forbid(unsafe_code)]

use ntd_capsule::{sha256, CapsuleKind, CapsuleView, Digest};

use crate::PortabilityError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum AssetClass {
    Model = 1,
    Capability = 2,
    Memory = 3,
    State = 4,
}

impl TryFrom<u8> for AssetClass {
    type Error = PortabilityError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Model),
            2 => Ok(Self::Capability),
            3 => Ok(Self::Memory),
            4 => Ok(Self::State),
            _ => Err(PortabilityError::InvalidAsset),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortableAsset {
    pub class: AssetClass,
    pub asset_id: String,
    pub version: u32,
    pub capsule: Vec<u8>,
    pub digest: Digest,
}

impl PortableAsset {
    pub fn from_capsule(
        class: AssetClass,
        asset_id: impl Into<String>,
        version: u32,
        capsule: Vec<u8>,
    ) -> Result<Self, PortabilityError> {
        let asset_id = asset_id.into().trim().to_owned();
        if asset_id.is_empty() || version == 0 || capsule.is_empty() {
            return Err(PortabilityError::InvalidAsset);
        }
        let view = CapsuleView::read(&capsule)
            .map_err(|error| PortabilityError::InvalidCapsule(format!("{error:?}")))?;
        match class {
            AssetClass::Model | AssetClass::Capability => {
                if !matches!(view.kind, CapsuleKind::Full | CapsuleKind::Thin) {
                    return Err(PortabilityError::InvalidAsset);
                }
            }
            AssetClass::Memory | AssetClass::State => {
                if view.kind != CapsuleKind::State {
                    return Err(PortabilityError::InvalidAsset);
                }
            }
        }
        Ok(Self {
            class,
            asset_id,
            version,
            digest: sha256(&capsule),
            capsule,
        })
    }

    pub fn is_stateful(&self) -> bool {
        matches!(self.class, AssetClass::Memory | AssetClass::State)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestoredSet {
    pub assets: Vec<PortableAsset>,
}

impl RestoredSet {
    pub fn asset(&self, class: AssetClass, asset_id: &str) -> Option<&PortableAsset> {
        self.assets
            .iter()
            .find(|asset| asset.class == class && asset.asset_id == asset_id)
    }
}
