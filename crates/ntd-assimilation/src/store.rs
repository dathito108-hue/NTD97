#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use crate::{
    verify_native_package, verify_native_package_with_shards, AssetKind, AssimilationError,
    FileTensorShardStore, NativePackage,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommitReceipt {
    pub asset_id: String,
    pub version: u32,
    pub kind: AssetKind,
}

#[derive(Debug, Clone)]
pub struct NativeAssetStore {
    trusted_verify_key: [u8; 32],
    versions: BTreeMap<String, BTreeMap<u32, NativePackage>>,
    active: BTreeMap<String, u32>,
}

impl NativeAssetStore {
    pub fn new(trusted_verify_key: [u8; 32]) -> Self {
        Self {
            trusted_verify_key,
            versions: BTreeMap::new(),
            active: BTreeMap::new(),
        }
    }

    pub fn commit_batch(
        &mut self,
        packages: Vec<NativePackage>,
    ) -> Result<Vec<CommitReceipt>, AssimilationError> {
        if packages.is_empty() {
            return Err(AssimilationError::InvalidPackage);
        }

        let mut staged_versions = self.versions.clone();
        let mut staged_active = self.active.clone();
        let mut receipts = Vec::new();

        for package in packages {
            verify_native_package(&package, &self.trusted_verify_key)?;
            let versions = staged_versions.entry(package.asset_id.clone()).or_default();
            if versions.contains_key(&package.version)
                || versions
                    .keys()
                    .next_back()
                    .is_some_and(|latest| package.version <= *latest)
            {
                return Err(AssimilationError::VersionConflict);
            }
            let receipt = CommitReceipt {
                asset_id: package.asset_id.clone(),
                version: package.version,
                kind: package.kind,
            };
            staged_active.insert(package.asset_id.clone(), package.version);
            versions.insert(package.version, package);
            receipts.push(receipt);
        }

        self.versions = staged_versions;
        self.active = staged_active;
        Ok(receipts)
    }

    pub fn commit_batch_with_shards(
        &mut self,
        packages: Vec<NativePackage>,
        shard_store: &FileTensorShardStore,
    ) -> Result<Vec<CommitReceipt>, AssimilationError> {
        if packages.is_empty() {
            return Err(AssimilationError::InvalidPackage);
        }

        let mut staged_versions = self.versions.clone();
        let mut staged_active = self.active.clone();
        let mut receipts = Vec::new();

        for package in packages {
            verify_native_package_with_shards(
                &package,
                &self.trusted_verify_key,
                shard_store,
            )?;
            let versions = staged_versions.entry(package.asset_id.clone()).or_default();
            if versions.contains_key(&package.version)
                || versions
                    .keys()
                    .next_back()
                    .is_some_and(|latest| package.version <= *latest)
            {
                return Err(AssimilationError::VersionConflict);
            }
            let receipt = CommitReceipt {
                asset_id: package.asset_id.clone(),
                version: package.version,
                kind: package.kind,
            };
            staged_active.insert(package.asset_id.clone(), package.version);
            versions.insert(package.version, package);
            receipts.push(receipt);
        }

        self.versions = staged_versions;
        self.active = staged_active;
        Ok(receipts)
    }

    pub fn active(&self, asset_id: &str) -> Result<&NativePackage, AssimilationError> {
        let version = self
            .active
            .get(asset_id)
            .ok_or_else(|| AssimilationError::MissingAsset(asset_id.into()))?;
        self.version(asset_id, *version)
    }

    pub fn version(
        &self,
        asset_id: &str,
        version: u32,
    ) -> Result<&NativePackage, AssimilationError> {
        self.versions
            .get(asset_id)
            .and_then(|versions| versions.get(&version))
            .ok_or_else(|| AssimilationError::MissingVersion {
                asset_id: asset_id.into(),
                version,
            })
    }

    pub fn rollback(
        &mut self,
        asset_id: &str,
        version: u32,
    ) -> Result<CommitReceipt, AssimilationError> {
        let package = self.version(asset_id, version)?.clone();
        verify_native_package(&package, &self.trusted_verify_key)?;
        self.active.insert(asset_id.to_owned(), version);
        Ok(CommitReceipt {
            asset_id: asset_id.to_owned(),
            version,
            kind: package.kind,
        })
    }

    pub fn rollback_with_shards(
        &mut self,
        asset_id: &str,
        version: u32,
        shard_store: &FileTensorShardStore,
    ) -> Result<CommitReceipt, AssimilationError> {
        let package = self.version(asset_id, version)?.clone();
        verify_native_package_with_shards(
            &package,
            &self.trusted_verify_key,
            shard_store,
        )?;
        self.active.insert(asset_id.to_owned(), version);
        Ok(CommitReceipt {
            asset_id: asset_id.to_owned(),
            version,
            kind: package.kind,
        })
    }

    pub fn versions(&self, asset_id: &str) -> Vec<u32> {
        self.versions
            .get(asset_id)
            .map(|versions| versions.keys().copied().collect())
            .unwrap_or_default()
    }

    pub fn asset_count(&self) -> usize {
        self.versions.len()
    }
}
