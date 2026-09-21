#![forbid(unsafe_code)]

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GeneralAgentScorecard {
    pub attempted_steps: u64,
    pub committed_steps: u64,
    pub recovered_failures: u64,
    pub unrecovered_failures: u64,
    pub security_denials: u64,
}

impl GeneralAgentScorecard {
    pub fn reliability_permille(self) -> u16 {
        if self.attempted_steps == 0 {
            return 0;
        }
        let value = self
            .committed_steps
            .saturating_mul(1000)
            .checked_div(self.attempted_steps)
            .unwrap_or(0)
            .min(1000);
        u16::try_from(value).unwrap_or(1000)
    }

    pub fn recovery_permille(self) -> u16 {
        let failures = self
            .recovered_failures
            .saturating_add(self.unrecovered_failures);
        if failures == 0 {
            return 1000;
        }
        let value = self
            .recovered_failures
            .saturating_mul(1000)
            .checked_div(failures)
            .unwrap_or(0)
            .min(1000);
        u16::try_from(value).unwrap_or(1000)
    }
}
