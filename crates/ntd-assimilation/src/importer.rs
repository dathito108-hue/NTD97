#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    AssimilationError, Discovery, NativeCandidate, SourcePackage,
};

pub trait SourceImporter {
    fn id(&self) -> &str;

    fn media_types(&self) -> &[&str];

    fn discover(&self, source: &SourcePackage) -> Result<Discovery, AssimilationError>;

    fn import(&self, source: &SourcePackage) -> Result<NativeCandidate, AssimilationError>;
}

#[derive(Default)]
pub struct ImporterRegistry {
    importers: BTreeMap<String, Box<dyn SourceImporter>>,
    media_types: BTreeMap<String, String>,
}

impl ImporterRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<I>(&mut self, importer: I) -> Result<(), AssimilationError>
    where
        I: SourceImporter + 'static,
    {
        let id = importer.id().trim().to_owned();
        if id.is_empty() || self.importers.contains_key(&id) {
            return Err(AssimilationError::DuplicateImporter(id));
        }

        let mut local = BTreeSet::new();
        for media_type in importer.media_types() {
            let media_type = media_type.trim();
            if media_type.is_empty() || !local.insert(media_type.to_owned()) {
                return Err(AssimilationError::DuplicateMediaType(media_type.into()));
            }
            if self.media_types.contains_key(media_type) {
                return Err(AssimilationError::DuplicateMediaType(media_type.into()));
            }
        }
        if local.is_empty() {
            return Err(AssimilationError::InvalidSource);
        }

        for media_type in local {
            self.media_types.insert(media_type, id.clone());
        }
        self.importers.insert(id, Box::new(importer));
        Ok(())
    }

    pub fn discover(&self, source: &SourcePackage) -> Result<Discovery, AssimilationError> {
        let importer = self.importer_for(source)?;
        let discovery = importer.discover(source)?;
        discovery.validate()?;
        if discovery.importer_id != importer.id() {
            return Err(AssimilationError::ImportRejected(
                "importer identity mismatch".into(),
            ));
        }
        Ok(discovery)
    }

    pub fn import(&self, source: &SourcePackage) -> Result<NativeCandidate, AssimilationError> {
        let importer = self.importer_for(source)?;
        let candidate = importer.import(source)?;
        candidate.validate()?;
        Ok(candidate)
    }

    pub fn importer_id_for(&self, media_type: &str) -> Result<&str, AssimilationError> {
        self.media_types
            .get(media_type)
            .map(String::as_str)
            .ok_or_else(|| AssimilationError::MissingImporter(media_type.into()))
    }

    fn importer_for(&self, source: &SourcePackage) -> Result<&dyn SourceImporter, AssimilationError> {
        let id = self.importer_id_for(&source.media_type)?;
        self.importers
            .get(id)
            .map(Box::as_ref)
            .ok_or_else(|| AssimilationError::MissingImporter(source.media_type.clone()))
    }
}
