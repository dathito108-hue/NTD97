# M10 Physical Device Evidence Collection

Major Block J cannot be closed from GitHub Actions or an emulator alone. The canonical evidence gate requires distinct physical Android devices representing the 4 GB, 8 GB and 12 GB memory bands.

## Privacy boundary

The collector does not read or store IMEI, Android ID, serial number, phone number, account data, contacts, files or location.

Device uniqueness is represented only by:

```text
SHA-256(Build.FINGERPRINT)
```

The raw Android build fingerprint is not written to the evidence record.

## What the collector measures

The debug-only `NtdPhysicalEvidenceActivity` records:

- RAM band: `mobile-4gb`, `mobile-8gb` or `mobile-12gb`;
- total physical RAM reported by Android;
- CI build revision embedded in the APK;
- p95 latency of a deterministic NTD97 native tiled-MatMul workload;
- energy per successful workload from Android BatteryManager when supported;
- native workload reliability;
- MCS97 continuity/recovery probe success rate;
- the sovereignty-audit bit embedded by Android CI after the dependency audit passes.

The benchmark runs on a background thread for at least 5 seconds and at least 32 samples, with a bounded ceiling of 8192 samples.

## Emulator rejection

The collector is intended only for physical-device evidence.

Both the Android activity and the host collection script reject common emulator signatures. The host script also checks `ro.kernel.qemu`.

This is a practical local validation guard, not cryptographic remote hardware attestation.

## Energy fail-closed behavior

Run the collector with the phone unplugged and thermally stable.

The preferred source is `BATTERY_PROPERTY_ENERGY_COUNTER`. If unavailable, the collector may use charge-counter × battery-voltage as an energy estimate.

If the phone is charging, the energy source is unavailable or changes, or no positive discharge delta is observed, the evidence stores the maximum `u64` value for energy-per-task. Such a record cannot pass an M10 energy target.

## APK provenance

Use the APK artifact produced by the Android CI workflow for the exact tested revision.

CI performs the no-third-party-AI dependency audit before assembly and embeds:

- `NTD_GIT_SHA`;
- `NTD_SOVEREIGNTY_AUDIT_PASSED=true`.

A locally assembled APK defaults to `local-unattested` and `false`.

## Collect from a real Android phone

Requirements:

- USB debugging enabled and authorized;
- Android platform tools / `adb` available;
- phone unplugged from charging power during the measurement. If USB supplies power, use a setup that permits ADB without charging or expect the energy metric to fail closed.

Run:

```bash
bash .github/scripts/collect-physical-evidence.sh /path/to/app-debug.apk physical-evidence
```

The script installs the debug APK, launches the collector, waits for completion, and exports:

- a human-readable `.txt` summary;
- a canonical binary `.nde97` evidence record.

## NDE97 record

NDE97 contains:

- physical evidence class;
- p95 latency;
- energy/task;
- reliability and recovery rates;
- sovereignty-audit result;
- total RAM and profile;
- sample count;
- SHA-256 Android-build fingerprint;
- build revision;
- energy source.

A SHA-256 digest covers the encoded record. It detects accidental or post-collection byte modification; it is not a hardware attestation signature.

## Completion rule

CI/emulator evidence is explicitly insufficient.

The final M10 evidence matrix must contain passing `PhysicalDevice` records for all required representative profiles with distinct fingerprint hashes and must satisfy the declared latency, energy, reliability, recovery and sovereignty targets.

Until those physical records exist, Major Block J remains in progress.
