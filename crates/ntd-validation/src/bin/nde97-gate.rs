#![forbid(unsafe_code)]

use std::{env, fs, path::PathBuf, process::ExitCode};

use ntd_validation::{
    decode_physical_evidence, evaluate_physical_records, PhysicalEvidenceRecord, ValidationTargets,
};

fn main() -> ExitCode {
    let mut args = env::args_os();
    let program = args.next().unwrap_or_else(|| "nde97-gate".into());
    let Some(expected_revision) = args.next() else {
        print_usage(&program);
        return ExitCode::from(2);
    };
    let Some(expected_revision) = expected_revision.to_str() else {
        eprintln!("expected revision must be UTF-8");
        return ExitCode::from(2);
    };

    let paths = args.map(PathBuf::from).collect::<Vec<_>>();
    if paths.is_empty() {
        print_usage(&program);
        return ExitCode::from(2);
    }

    let mut records = Vec::<PhysicalEvidenceRecord>::new();
    for path in &paths {
        let bytes = match fs::read(path) {
            Ok(bytes) => bytes,
            Err(error) => {
                eprintln!("{}: read failed: {error}", path.display());
                return ExitCode::from(2);
            }
        };
        let record = match decode_physical_evidence(&bytes) {
            Ok(record) => record,
            Err(error) => {
                eprintln!("{}: invalid NDE97 record: {error:?}", path.display());
                return ExitCode::from(2);
            }
        };
        println!(
            "{} profile={} revision={} p95_ns={} energy_uj={} reliability={} recovery={} audit={} source={}",
            path.display(),
            record.evidence.profile,
            record.build_revision,
            record.evidence.p95_latency_nanos,
            record.evidence.energy_per_task_microjoules,
            record.evidence.reliability_permille,
            record.evidence.recovery_permille,
            record.evidence.sovereignty_audit_passed,
            record.energy_source,
        );
        records.push(record);
    }

    let targets = ValidationTargets::m10_reference();
    let report = evaluate_physical_records(&records, &targets, expected_revision);
    if report.accepted {
        println!(
            "M10 physical evidence gate PASS profiles={}",
            report.accepted_profiles.join(",")
        );
        ExitCode::SUCCESS
    } else {
        for failure in &report.failures {
            eprintln!("M10 gate failure: {failure:?}");
        }
        ExitCode::FAILURE
    }
}

fn print_usage(program: &std::ffi::OsStr) {
    eprintln!(
        "usage: {} <40-char-build-revision> <record.nde97> [record.nde97 ...]",
        PathBuf::from(program).display()
    );
}
