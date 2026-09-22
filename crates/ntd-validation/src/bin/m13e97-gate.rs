#![forbid(unsafe_code)]

use std::{env, fs, path::PathBuf, process::ExitCode};

use ntd_validation::{decode_m13_hardware_evidence, M13_REQUIRED_GATE_MASK};

fn main() -> ExitCode {
    let mut args = env::args_os();
    let program = args.next().unwrap_or_else(|| "m13e97-gate".into());
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

    let mut accepted = 0usize;
    for path in &paths {
        let bytes = match fs::read(path) {
            Ok(bytes) => bytes,
            Err(error) => {
                eprintln!("{}: read failed: {error}", path.display());
                return ExitCode::from(2);
            }
        };
        let record = match decode_m13_hardware_evidence(&bytes) {
            Ok(record) => record,
            Err(error) => {
                eprintln!("{}: invalid M13E97 record: {error:?}", path.display());
                return ExitCode::from(2);
            }
        };
        println!(
            "{} revision={} fingerprint={} gates=0x{:04x}/0x{:04x} verified_actions={} process_death_reconfirmed={} sovereignty_audit={}",
            path.display(),
            record.build_revision,
            record.device_fingerprint,
            record.gate_mask,
            M13_REQUIRED_GATE_MASK,
            record.verified_action_count,
            record.process_death_reconfirmed,
            record.sovereignty_audit_passed,
        );
        if !record.accepted_for_revision(expected_revision) {
            eprintln!(
                "{}: M13 hardware acceptance record does not meet the canonical gate",
                path.display()
            );
            return ExitCode::FAILURE;
        }
        accepted += 1;
    }

    println!("M13 hardware evidence gate PASS records={accepted}");
    ExitCode::SUCCESS
}

fn print_usage(program: &std::ffi::OsStr) {
    eprintln!(
        "usage: {} <40-char-build-revision> <record.m13e97> [record.m13e97 ...]",
        PathBuf::from(program).display()
    );
}
