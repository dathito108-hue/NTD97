#![forbid(unsafe_code)]

use std::time::Instant;

pub trait EnergySampler {
    fn sample_microjoules(&mut self) -> Option<u64>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NoEnergySampler;

impl EnergySampler for NoEnergySampler {
    fn sample_microjoules(&mut self) -> Option<u64> {
        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceSpan {
    pub label: String,
    pub elapsed_nanos: u64,
    pub energy_microjoules: Option<u64>,
}

pub struct TraceRecorder<S> {
    sampler: S,
    spans: Vec<TraceSpan>,
}

impl<S> TraceRecorder<S>
where
    S: EnergySampler,
{
    pub fn new(sampler: S) -> Self {
        Self {
            sampler,
            spans: Vec::new(),
        }
    }

    pub fn measure<T, F>(&mut self, label: impl Into<String>, operation: F) -> T
    where
        F: FnOnce() -> T,
    {
        let before_energy = self.sampler.sample_microjoules();
        let started = Instant::now();
        let result = operation();
        let elapsed_nanos = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let after_energy = self.sampler.sample_microjoules();
        let energy_microjoules = match (before_energy, after_energy) {
            (Some(before), Some(after)) => after.checked_sub(before),
            _ => None,
        };
        self.spans.push(TraceSpan {
            label: label.into(),
            elapsed_nanos,
            energy_microjoules,
        });
        result
    }

    pub fn spans(&self) -> &[TraceSpan] {
        &self.spans
    }

    pub fn percentile_nanos(&self, permille: u16) -> Option<u64> {
        if self.spans.is_empty() || permille > 1000 {
            return None;
        }
        let mut values = self
            .spans
            .iter()
            .map(|span| span.elapsed_nanos)
            .collect::<Vec<_>>();
        values.sort_unstable();
        let last = values.len() - 1;
        let scaled = last
            .checked_mul(usize::from(permille))
            .unwrap_or(usize::MAX);
        let index = scaled.div_ceil(1000).min(last);
        values.get(index).copied()
    }

    pub fn total_energy_microjoules(&self) -> Option<u64> {
        if self.spans.is_empty()
            || self
                .spans
                .iter()
                .any(|span| span.energy_microjoules.is_none())
        {
            return None;
        }
        Some(
            self.spans
                .iter()
                .filter_map(|span| span.energy_microjoules)
                .fold(0u64, u64::saturating_add),
        )
    }

    pub fn into_sampler(self) -> S {
        self.sampler
    }
}
