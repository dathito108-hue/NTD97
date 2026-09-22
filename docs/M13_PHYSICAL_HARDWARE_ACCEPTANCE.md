# M13 Physical Hardware Acceptance

M13 is not complete from GitHub Actions or emulator acceptance alone. The remaining exit gates require a representative physical Android phone, real user grants, a real third-party accessibility target, a reachable PCF97 desktop agent and a real HTTPS upload endpoint.

This harness extends the existing physical-evidence workflow. It does **not** introduce a test adapter or side-effect shortcut.

## What the harness proves

The debug-only `NtdM13HardwareEvidenceActivity` drives the same native chat API used by the normal UI:

`submitChat -> approval -> ActionFabric -> production adapter -> verifier -> verified synthesis`.

Every external write is resolved through `resolveChatApproval(true)`. The harness can auto-resolve those approvals only when it is launched with the explicit debug intent flag `approve_external=true`.

The physical sequence requires verified production actions for:

- provider-independent WebSearch;
- browser observe/navigation on public HTTPS;
- SAF user-granted file write and read through the persisted `shared` tree;
- clipboard write;
- exact-package launch plus third-party accessibility `set_text`;
- authenticated PCF97 observe and artifact write against a real paired desktop agent;
- resumable HTTPS artifact upload across a real Android process death.

The sequence uses the same sovereign runtime throughout. No mock capability adapter is accepted by the final record.

## Process-death upload gate

Artifact upload is intentionally stopped at the durable pre-network `ACTION_CHECKPOINTED` boundary. The collector then:

1. persists the NCS97 conversation checkpoint;
2. records both Android PID and an in-process random instance nonce;
3. force-stops the app;
4. verifies that the process exited;
5. launches a new process;
6. restores the NCS97 checkpoint;
7. requires the first restored action event to be `APPROVAL_REQUIRED`;
8. explicitly approves the reconfirmation;
9. resumes and verifies the HTTPS PUT.

The evidence cannot mark the process-death gate complete if PID or process-instance nonce did not change, if restore auto-replays the write, or if verified action evidence is missing.

## Prerequisites on the phone

Use the canonical CI debug APK for the exact main revision being validated. The APK must embed:

- a 40-character `NTD_GIT_SHA`;
- `NTD_SOVEREIGNTY_AUDIT_PASSED=true`.

Before running the collector, configure the production app state:

1. **WebSearch** — configure a working Generic JSON or OpenSearch profile. Authenticated profiles are allowed.
2. **SAF** — use the normal Storage grant UI to persist a writable tree as alias `shared`.
3. **Accessibility** — manually enable NTD97 Accessibility in Android system settings. Prepare a real third-party app/package with one known editable `view_id`.
4. **Paired PC** — provision a real PCF97 desktop agent through the Paired PC UI and keep it online.
5. **Upload** — prepare a public HTTPS endpoint that accepts idempotent PUT and returns 2xx.

The collector never self-enables AccessibilityService, creates a fake SAF grant, or synthesizes a paired-PC server.

## Required host variables

Set these before running the ADB collector:

```bash
export NTD97_M13_SEARCH_QUERY='NTD97 mobile'
export NTD97_M13_BROWSER_URL='https://example.com/'
export NTD97_M13_STORAGE_PATH='ntd97/m13-hardware.txt'
export NTD97_M13_UPLOAD_URL='https://your-public-endpoint.example/upload/object'
export NTD97_M13_ACCESSIBILITY_PACKAGE='com.example.target'
export NTD97_M13_ACCESSIBILITY_VIEW_ID='com.example.target:id/input'
export NTD97_M13_PC_ALIAS='workstation'
export NTD97_M13_PC_SURFACE='system'
export NTD97_M13_PC_PATH='shared/ntd97-m13.txt'
```

String extras are Base64 encoded by the host script before they cross the ADB shell boundary, so URL query parameters and other shell-sensitive characters are not interpreted by the remote shell.

## Collect evidence

Run from a checkout whose HEAD is the expected canonical revision:

```bash
bash .github/scripts/collect-m13-hardware-evidence.sh \
  /path/to/app-debug.apk \
  m13-hardware-evidence \
  <40-char-canonical-main-sha>
```

The script:

- rejects `ro.kernel.qemu=1`;
- installs the APK with `-r` so existing grants/configuration survive;
- checks that NTD97 AccessibilityService is already user-enabled;
- launches phase A;
- waits for `status=awaiting-process-death`;
- verifies the APK build revision;
- force-stops NTD97 and confirms the process is gone;
- launches phase B;
- waits for `status=ok`;
- exports the text summary and binary `.m13e97`;
- runs the canonical `m13e97-gate` locally.

## M13E97 record

M13E97 stores only acceptance metadata:

- canonical build revision;
- SHA-256 of Android `Build.FINGERPRINT`;
- required gate bitmask;
- total verifier-committed action count;
- process-death/reconfirm result;
- sovereignty-audit bit;
- SHA-256 of a per-run random nonce.

It intentionally does **not** contain prompts, WebSearch credentials, SAF contents, clipboard text, accessibility-entered text, paired-PC payloads or raw ActionFabric receipts.

A SHA-256 digest protects record integrity. As with NDE97, this is not cryptographic hardware attestation; emulator rejection and physical-device collection are practical acceptance guards.

## Verify a record manually

```bash
cargo run -p ntd-validation --bin m13e97-gate -- \
  <40-char-canonical-main-sha> \
  m13-hardware-evidence/record.m13e97
```

The gate requires all ten hardware gates, at least ten verifier-committed actions, a real process-death reconfirm, a matching build revision and a passing sovereignty audit.

## Completion rule

GitHub Actions/emulator results prove the implementation and fail-closed contracts, but they cannot close M13.

A passing representative physical-phone M13E97 record is a **necessary mixed-core gate, not a sufficient M13 completion certificate**. It must use the intended real SAF provider, real accessibility target, real PCF97 peer and real HTTPS upload service.

M13 remains **IN PROGRESS** until that record exists **and** the additional representative-phone edge cases listed in the canonical roadmap (credential rotation/restart, browser/session edge cases, SAF revoke/reselect, accessibility enable/disable/OEM behavior, and PC reconnect/re-pair) are also evidenced.
