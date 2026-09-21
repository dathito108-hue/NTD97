#![forbid(unsafe_code)]

use ntd_runtime::{PagedByteReader, PrefixCache, SliceByteRegion};

use crate::ValidationError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixReuseReport {
    pub reusable_tokens: usize,
    pub requested_tokens: usize,
    pub reuse_permille: u16,
}

pub fn analyze_prefix_reuse(cache: &PrefixCache, requested: &[u32]) -> PrefixReuseReport {
    let reusable_tokens = cache.reusable_prefix_len(requested);
    let reuse_permille = if requested.is_empty() {
        0
    } else {
        u16::try_from(
            reusable_tokens
                .saturating_mul(1000)
                .checked_div(requested.len())
                .unwrap_or(0)
                .min(1000),
        )
        .unwrap_or(1000)
    };
    PrefixReuseReport {
        reusable_tokens,
        requested_tokens: requested.len(),
        reuse_permille,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagingReport {
    pub logical_bytes: usize,
    pub page_bytes: u64,
    pub page_count: usize,
}

pub fn verify_paged_round_trip(
    bytes: &[u8],
    page_bytes: u64,
) -> Result<PagingReport, ValidationError> {
    let reader = PagedByteReader::new(SliceByteRegion::new(bytes), page_bytes)
        .map_err(|error| ValidationError::MobileCompute(format!("{error:?}")))?;
    let mut restored = Vec::with_capacity(bytes.len());
    for index in 0..reader.page_count() {
        restored.extend_from_slice(reader.page(index).ok_or(ValidationError::PagingMismatch)?);
    }
    if restored != bytes {
        return Err(ValidationError::PagingMismatch);
    }
    Ok(PagingReport {
        logical_bytes: bytes.len(),
        page_bytes,
        page_count: reader.page_count(),
    })
}
