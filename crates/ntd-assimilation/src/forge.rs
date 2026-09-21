#![forbid(unsafe_code)]

use std::collections::BTreeSet;

use crate::{
    build_native_package, AssimilationError, AssimilationIdentity, CommitReceipt, Discovery,
    ForgeSandbox, ImporterRegistry, NativeAssetStore, SourcePackage,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForgePolicy {
    pub allowed_licenses: BTreeSet<String>,
    pub max_source_bytes: usize,
}

impl ForgePolicy {
    pub fn new(
        allowed_licenses: impl IntoIterator<Item = String>,
        max_source_bytes: usize,
    ) -> Result<Self, AssimilationError> {
        let allowed_licenses = allowed_licenses
            .into_iter()
            .map(|license| license.trim().to_owned())
            .filter(|license| !license.is_empty())
            .collect::<BTreeSet<_>>();
        if allowed_licenses.is_empty() || max_source_bytes == 0 {
            return Err(AssimilationError::InvalidSource);
        }
        Ok(Self {
            allowed_licenses,
            max_source_bytes,
        })
    }
}

pub struct CapabilityForge<S> {
    importers: ImporterRegistry,
    sandbox: S,
    identity: AssimilationIdentity,
    policy: ForgePolicy,
}

impl<S> CapabilityForge<S>
where
    S: ForgeSandbox,
{
    pub fn new(
        importers: ImporterRegistry,
        sandbox: S,
        identity: AssimilationIdentity,
        policy: ForgePolicy,
    ) -> Self {
        Self {
            importers,
            sandbox,
            identity,
            policy,
        }
    }

    pub fn discover(&self, source: &SourcePackage) -> Result<Discovery, AssimilationError> {
        self.validate_source(source)?;
        self.importers.discover(source)
    }

    pub fn assimilate(
        &mut self,
        source: &SourcePackage,
        store: &mut NativeAssetStore,
    ) -> Result<CommitReceipt, AssimilationError> {
        self.validate_source(source)?;
        let discovery = self.importers.discover(source)?;
        let candidate = self.importers.import(source)?;
        if discovery.asset_kind != candidate.kind()
            || discovery.asset_id_hint != candidate.asset_id()
        {
            return Err(AssimilationError::ImportRejected(
                "discovery/import mismatch".into(),
            ));
        }

        let report = self.sandbox.validate(&candidate)?;
        if !report.isolated || report.network_used || report.external_write_used {
            return Err(AssimilationError::SandboxNotIsolated);
        }

        let package = build_native_package(
            &candidate,
            &source.provenance,
            &discovery.importer_id,
            &report,
            &self.identity,
        )?;
        let mut receipts = store.commit_batch(vec![package])?;
        receipts.pop().ok_or(AssimilationError::InvalidPackage)
    }

    pub fn identity(&self) -> &AssimilationIdentity {
        &self.identity
    }

    fn validate_source(&self, source: &SourcePackage) -> Result<(), AssimilationError> {
        source.provenance.validate()?;
        if source.payload.is_empty() || source.media_type.trim().is_empty() {
            return Err(AssimilationError::InvalidSource);
        }
        if source.payload.len() > self.policy.max_source_bytes {
            return Err(AssimilationError::SourceTooLarge);
        }
        if !self
            .policy
            .allowed_licenses
            .contains(source.provenance.license.spdx_id.trim())
        {
            return Err(AssimilationError::LicenseDenied(
                source.provenance.license.spdx_id.clone(),
            ));
        }
        if ntd_capsule::sha256(&source.payload) != source.provenance.source_digest {
            return Err(AssimilationError::InvalidProvenance);
        }
        Ok(())
    }
}
