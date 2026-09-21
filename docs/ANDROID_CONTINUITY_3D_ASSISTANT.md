# Android Continuity + Interactive 3D Assistant

Status: Major Block F complete; canonical Android build and API 35 lifecycle/reboot validation PASS.

This block binds the existing SIK97 cognition state and TAF97 action journal to a mobile continuity shell and a first-class 3D assistant surface without moving Android APIs into `ntd-runtime`.

## 1. Dependency boundary

```text
ntd-runtime
   ^
   |
ntd-mobile-shell
   ^
   |
platform/android
```

`ntd-mobile-shell` contains portable continuity, wake-policy, voice/media and avatar-state contracts. `platform/android` owns Android lifecycle, storage, scheduling, notifications, overlays and OpenGL ES rendering.

## 2. MCS97 mobile continuity bundle

`MCS97\0` is the deterministic mobile continuity envelope for:

- stable `CognitiveIdentity`;
- SIK97 cognitive checkpoint;
- TAF97 action checkpoint;
- canonical capability descriptor snapshot with pinned versions/scopes/side-effect metadata;
- continuity state;
- wake reason;
- monotonic checkpoint sequence;
- pending approval identity;
- retry/backoff metadata.

Current format:

```text
major = 0
minor = 1
```

Build and restore validate the nested SIK97 and TAF97 checkpoints and cross-check every action plan against an existing cognitive task. Capability descriptors are carried inside MCS97, so reboot restore does not require Android to reconstruct a registry by hand. A pending approval must match an actual task/plan/action/capability tuple.

Authority grants and platform adapters are deliberately not serialized.

## 3. Logical 24/7 continuity

The mobile state machine uses:

- Interactive;
- ActiveExecution;
- Checkpointed;
- SuspendedByOs;
- WaitingCondition;
- WaitingApproval;
- Reconstructing;
- VerifyingResume;
- Completed;
- FailedRecoverable.

Wake reasons include user interaction, foreground execution, scheduled work, retry, reboot, approval resolution, network availability, charging and manual recovery.

Logical continuity means durable state can reconstruct the same identity/task after UI exit, process death or reboot. It does not claim unrestricted background CPU residency.

## 4. OS-aware wake policy

`choose_platform_directive` maps MCS97 state plus resource state and platform context to one of:

- keep interactive;
- foreground execution;
- persistent scheduled work;
- approval surface;
- checkpoint-and-suspend;
- verify resume;
- no eligible work.

Critical thermal pressure or near-empty battery forces checkpoint/suspend rather than continued work. Retry windows remain durable across wakes.

## 5. Process-death/reboot acceptance

The end-to-end test performs:

```text
CognitiveTask
  -> action 1 ExternalWrite commits once
  -> action 2 WebSearch suspends
  -> MCS97 checkpoint
  -> process object destroyed
  -> reboot-style decode/restore
  -> reattach only action-2 adapter
  -> resume action 2
  -> Completed
```

The committed external write must remain at exactly one execution. This proves MCS97 + TAF97 preserve the action cursor and stable action identity rather than replaying completed work.

## 6. Android storage and reboot bootstrap

`NtdContinuityStore` uses Android `AtomicFile` in device-protected no-backup storage. The Android shell declares reboot reception and schedules eligible durable work through platform `JobScheduler`.

`NtdBootReceiver` never executes the agent directly. It only schedules the canonical restore path.

## 7. Foreground and scheduled execution

`NtdContinuityJobService` is the deferred wake path.

`NtdForegroundService` is used for eligible user-visible active work. Both enter through `NtdSessionController.restore`, so UI, scheduled wake and foreground execution share one restore/verify gate.

If no local runtime host is attached, durable work remains stored and scheduled work is rescheduled. The shell never reports a successful restore without a local host verifying MCS97.

## 8. Local runtime host boundary

`NtdRuntimeHost` remains the local binding interface between the Android shell and packaged NTD97 runtime.

The canonical implementation is `NtdNativeRuntimeHost`, which loads `libntd97_android.so` and crosses a narrow JNI boundary into the `ntd-android-bridge` Rust cdylib. That bridge depends inward on `ntd-mobile-shell` and `ntd-runtime`, so MCS97 decode/verify/restore and avatar state remain native NTD97 logic rather than a second Java cognition implementation. `NtdRuntimeBootstrap` attaches this concrete local host directly; there is no ServiceLoader dependency, hosted runtime, third-party AI backend or cloud fallback.

The host owns:

- restore + verify;
- checkpoint production;
- approval resolution;
- avatar-state export;
- local PCM ingress/egress;
- Android resource snapshot handoff for RAM/battery/charging/thermal-aware rendering.

## 9. Notifications and approval

Execution and approval use separate notification channels.

Approval notifications contain only generic discoverability text and use private lock-screen visibility. Capability/rationale details remain inside the app surface.

Approve/deny is routed back to the local runtime host and followed by a durable checkpoint.

## 10. Interactive 3D assistant

`AvatarController` produces renderer-independent state:

- Idle / Active / Thinking / Acting / WaitingApproval / Sleeping / Recovering / Completed / Error;
- facial expression;
- gesture;
- gaze target;
- lip-sync amplitude + viseme;
- task/action progress;
- status text;
- render profile.

`NtdAvatarGLSurfaceView` is an Android OpenGL ES surface consuming that state. It supports touch rotation and gaze-driven pose changes. Lip amplitude affects the speaking pose.

The same avatar can render in-app or in the permission-gated floating overlay.

## 11. Thermal/battery rendering

Avatar render profile scales from 60 FPS / higher detail down to 5 FPS / minimal effects under critical thermal or battery pressure.

Rendering is therefore a consumer of mobile resource policy rather than a requirement for cognition.

## 12. Voice/media contract

`VoiceStateMachine` defines local input/output state, transcript handoff, speaking progress, amplitude and viseme state. It contains no cloud speech dependency.

The Android shell also includes a local PCM `AudioRecord` / `AudioTrack` bridge. Microphone PCM is delivered only to the packaged local runtime host and speaker PCM is pulled from that host. No Android SpeechRecognizer/TTS service is required for the sovereign baseline.

## 13. Android build and lifecycle validation

`platform/android` contains the application shell, manifest, atomic continuity store, JobScheduler bridge, foreground service, reboot receiver, notification/approval controller, in-app OpenGL ES avatar, floating overlay, resource sampler and concrete native runtime host.

Canonical CI now verifies:

- Rust `cargo fmt --all -- --check`;
- `cargo clippy --workspace --all-targets -- -D warnings`;
- `cargo test --workspace`;
- Android SDK 35 + NDK 27 setup;
- Rust native bridge cross-compilation for `arm64-v8a`, `armeabi-v7a` and `x86_64`;
- Gradle 8.9 / JDK 17 debug APK assembly;
- API 35 x86_64 emulator lifecycle validation.

The emulator gate confirms native host attachment, avatar JNI, PCM bridge, notification channels, foreground-service request, overlay service, persisted JobScheduler work, reboot and post-reboot cold start. The debug-only probe is excluded from release source sets.

This closes the M6 integration gate. A broader Android-version/device matrix, long-duration soak and performance/thermal validation remain intentionally assigned to M10 rather than being duplicated here.
