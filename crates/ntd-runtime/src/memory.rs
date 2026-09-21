#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum MemoryKind {
    Episodic = 1,
    Semantic = 2,
    Procedural = 3,
}

impl TryFrom<u8> for MemoryKind {
    type Error = MemoryError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Episodic),
            2 => Ok(Self::Semantic),
            3 => Ok(Self::Procedural),
            other => Err(MemoryError::InvalidKind(other)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryRecord {
    pub id: u64,
    pub kind: MemoryKind,
    pub content: String,
    pub tags: Vec<String>,
    pub importance: u16,
    pub created_tick: u64,
    pub last_recalled_tick: u64,
    pub recall_count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryQuery {
    pub text: String,
    pub kinds: Vec<MemoryKind>,
    pub tags: Vec<String>,
    pub limit: usize,
}

impl MemoryQuery {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            kinds: Vec::new(),
            tags: Vec::new(),
            limit: 8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryHit {
    pub id: u64,
    pub score: u64,
    pub record: MemoryRecord,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryError {
    EmptyContent,
    InvalidImportance(u16),
    InvalidKind(u8),
    Overflow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SovereignMemory {
    next_id: u64,
    records: BTreeMap<u64, MemoryRecord>,
}

impl Default for SovereignMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl SovereignMemory {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            records: BTreeMap::new(),
        }
    }

    pub fn from_parts(
        next_id: u64,
        records: BTreeMap<u64, MemoryRecord>,
    ) -> Result<Self, MemoryError> {
        if next_id == 0 {
            return Err(MemoryError::Overflow);
        }

        for (id, record) in &records {
            if *id == 0
                || *id != record.id
                || record.content.trim().is_empty()
                || record.importance > 1000
                || !record.tags.windows(2).all(|pair| pair[0] < pair[1])
            {
                return Err(MemoryError::Overflow);
            }
        }

        if let Some(max_id) = records.keys().next_back() {
            if next_id <= *max_id {
                return Err(MemoryError::Overflow);
            }
        }

        Ok(Self { next_id, records })
    }

    pub fn next_id(&self) -> u64 {
        self.next_id
    }

    pub fn records(&self) -> &BTreeMap<u64, MemoryRecord> {
        &self.records
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn store(
        &mut self,
        kind: MemoryKind,
        content: impl Into<String>,
        tags: Vec<String>,
        importance: u16,
        tick: u64,
    ) -> Result<u64, MemoryError> {
        let content = content.into();
        if content.trim().is_empty() {
            return Err(MemoryError::EmptyContent);
        }
        if importance > 1000 {
            return Err(MemoryError::InvalidImportance(importance));
        }

        let id = self.next_id;
        self.next_id = self.next_id.checked_add(1).ok_or(MemoryError::Overflow)?;

        let mut tags = tags;
        tags.sort();
        tags.dedup();

        self.records.insert(
            id,
            MemoryRecord {
                id,
                kind,
                content,
                tags,
                importance,
                created_tick: tick,
                last_recalled_tick: tick,
                recall_count: 0,
            },
        );
        Ok(id)
    }

    pub fn get(&self, id: u64) -> Option<&MemoryRecord> {
        self.records.get(&id)
    }

    pub fn forget(&mut self, id: u64) -> bool {
        self.records.remove(&id).is_some()
    }

    pub fn retrieve(
        &mut self,
        query: &MemoryQuery,
        tick: u64,
    ) -> Result<Vec<MemoryHit>, MemoryError> {
        if query.limit == 0 {
            return Ok(Vec::new());
        }

        let query_terms = normalized_terms(&query.text);
        let query_tags = query
            .tags
            .iter()
            .map(|tag| tag.to_ascii_lowercase())
            .collect::<BTreeSet<_>>();
        let kind_filter = query.kinds.iter().copied().collect::<BTreeSet<_>>();

        let mut ranked = self
            .records
            .values()
            .filter(|record| kind_filter.is_empty() || kind_filter.contains(&record.kind))
            .filter_map(|record| {
                score_record(record, &query_terms, &query_tags).map(|score| (record.id, score))
            })
            .collect::<Vec<_>>();

        ranked.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
        ranked.truncate(query.limit);

        let mut hits = Vec::with_capacity(ranked.len());
        for (id, score) in ranked {
            let record = self.records.get_mut(&id).ok_or(MemoryError::Overflow)?;
            record.last_recalled_tick = tick;
            record.recall_count = record.recall_count.saturating_add(1);
            hits.push(MemoryHit {
                id,
                score,
                record: record.clone(),
            });
        }
        Ok(hits)
    }
}

fn score_record(
    record: &MemoryRecord,
    query_terms: &BTreeSet<String>,
    query_tags: &BTreeSet<String>,
) -> Option<u64> {
    let content_terms = normalized_terms(&record.content);
    let term_matches = query_terms.intersection(&content_terms).count() as u64;

    let record_tags = record
        .tags
        .iter()
        .map(|tag| tag.to_ascii_lowercase())
        .collect::<BTreeSet<_>>();
    let tag_matches = query_tags.intersection(&record_tags).count() as u64;

    let has_query = !query_terms.is_empty() || !query_tags.is_empty();
    if has_query && term_matches == 0 && tag_matches == 0 {
        return None;
    }

    Some(
        term_matches
            .saturating_mul(1_000)
            .saturating_add(tag_matches.saturating_mul(500))
            .saturating_add(u64::from(record.importance))
            .saturating_add(u64::from(record.recall_count.min(100))),
    )
}

fn normalized_terms(text: &str) -> BTreeSet<String> {
    text.split_whitespace()
        .map(|term| {
            term.trim_matches(|ch: char| !ch.is_alphanumeric())
                .to_ascii_lowercase()
        })
        .filter(|term| !term.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retrieval_prefers_semantic_overlap_then_importance() {
        let mut memory = SovereignMemory::new();
        memory
            .store(
                MemoryKind::Semantic,
                "Paris is the capital of France",
                vec!["geography".into()],
                600,
                1,
            )
            .expect("store");
        memory
            .store(
                MemoryKind::Episodic,
                "Visited Paris during summer",
                vec!["travel".into()],
                200,
                2,
            )
            .expect("store");

        let hits = memory
            .retrieve(&MemoryQuery::new("capital France"), 3)
            .expect("retrieve");

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].record.kind, MemoryKind::Semantic);
    }

    #[test]
    fn tag_query_can_recall_without_text_overlap() {
        let mut memory = SovereignMemory::new();
        let id = memory
            .store(
                MemoryKind::Procedural,
                "verify output before commit",
                vec!["workflow".into()],
                500,
                1,
            )
            .expect("store");

        let mut query = MemoryQuery::new("");
        query.tags.push("workflow".into());

        let hits = memory.retrieve(&query, 2).expect("retrieve");
        assert_eq!(hits[0].id, id);
        assert_eq!(memory.get(id).expect("record").recall_count, 1);
    }

    #[test]
    fn forget_removes_record() {
        let mut memory = SovereignMemory::new();
        let id = memory
            .store(MemoryKind::Episodic, "temporary", Vec::new(), 1, 0)
            .expect("store");
        assert!(memory.forget(id));
        assert!(memory.is_empty());
    }
}
