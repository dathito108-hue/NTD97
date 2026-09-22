#!/usr/bin/env bash
set -euo pipefail

APK="platform/android/app/build/outputs/apk/debug/app-debug.apk"
PACKAGE="ai.ntd97.mobile"
PROBE_ACTIVITY="${PACKAGE}/.NtdLifecycleProbeActivity"
PHYSICAL_ACTIVITY="${PACKAGE}/.NtdPhysicalEvidenceActivity"
REAL_MODEL_ACTIVITY="${PACKAGE}/.NtdRealModelProbeActivity"
MAIN_ACTIVITY="${PACKAGE}/.MainActivity"

adb install -r "${APK}"
adb shell pm grant "${PACKAGE}" android.permission.POST_NOTIFICATIONS
adb shell pm grant "${PACKAGE}" android.permission.RECORD_AUDIO
adb shell appops set "${PACKAGE}" SYSTEM_ALERT_WINDOW allow

adb shell am start -W -n "${MAIN_ACTIVITY}"
test -n "$(adb shell pidof "${PACKAGE}" | tr -d '\r')"

adb shell am start -W -n "${PROBE_ACTIVITY}" --ez seed_reboot true
sleep 2

PROBE="$(adb shell run-as "${PACKAGE}" cat files/ntd97-lifecycle-probe.txt | tr -d '\r')"
printf '%s\n' "${PROBE}"

for REQUIRED in \
  build_attestation=ok \
  native_host=ok \
  avatar=ok \
  audio_bridge=ok \
  notification_channels=ok \
  foreground_request=ok \
  overlay_request=ok \
  persistent_job=ok \
  reboot_marker=ok
do
  printf '%s\n' "${PROBE}" | grep -Fxq "${REQUIRED}"
done

adb shell dumpsys activity services "${PACKAGE}" | grep -q "NtdOverlayService"

adb shell run-as "${PACKAGE}" rm -f files/ntd97-real-model-probe.txt || true
adb shell am start -W -n "${REAL_MODEL_ACTIVITY}" >/dev/null

for ATTEMPT in $(seq 1 120); do
  if adb shell run-as "${PACKAGE}" test -f files/ntd97-real-model-probe.txt; then
    break
  fi
  sleep 1
done

REAL_MODEL_PROBE="$(adb shell run-as "${PACKAGE}" cat files/ntd97-real-model-probe.txt | tr -d '\r')"
printf '%s\n' "${REAL_MODEL_PROBE}"

for REQUIRED in \
  signature=ok \
  activation=ok \
  android_real_model=PASS \
  web_authority=ok \
  web_fetch=ok \
  web_receipt=ok \
  chat_submit=ok \
  chat_stream=ok \
  chat_reasoning=ok \
  chat_reasoning_loop=ok \
  chat_action_planner=ok \
  chat_action_safety=ok \
  chat_governed_e2e=ok \
  chat_memory=ok \
  chat_restore=ok \
  chat_store=ok \
  chat_cancel=ok \
  chat_status=ok
do
  printf '%s\n' "${REAL_MODEL_PROBE}" | grep -Fxq "${REQUIRED}"
done

printf '%s\n' "${REAL_MODEL_PROBE}" | grep -Eq '^generated_token_count=[1-9][0-9]*$'
printf '%s\n' "${REAL_MODEL_PROBE}" | grep -Eq '^generated_token_ids=[0-9]+(,[0-9]+)*$'
printf '%s\n' "${REAL_MODEL_PROBE}" | grep -Eq '^generated_text_sha256=[0-9a-f]{64}

adb shell run-as "${PACKAGE}" rm -f \
  files/ntd97-device-evidence.nde97 \
  files/ntd97-device-evidence.txt || true
adb shell am start -W -n "${PHYSICAL_ACTIVITY}" >/dev/null

for ATTEMPT in $(seq 1 30); do
  if adb shell run-as "${PACKAGE}" test -f files/ntd97-device-evidence.txt; then
    break
  fi
  sleep 1
done

PHYSICAL_REJECTION="$(adb shell run-as "${PACKAGE}" cat files/ntd97-device-evidence.txt | tr -d '\r')"
printf '%s\n' "${PHYSICAL_REJECTION}"
printf '%s\n' "${PHYSICAL_REJECTION}" | grep -Fxq "status=emulator-rejected"

if adb shell run-as "${PACKAGE}" test -f files/ntd97-device-evidence.nde97; then
  echo "emulator unexpectedly produced PhysicalDevice evidence" >&2
  exit 1
fi

adb reboot
adb wait-for-device
for ATTEMPT in $(seq 1 90); do
  if [ "$(adb shell getprop sys.boot_completed | tr -d '\r')" = "1" ]; then
    break
  fi
  sleep 2
done

test "$(adb shell getprop sys.boot_completed | tr -d '\r')" = "1"

PACKAGE_READY=false
for ATTEMPT in $(seq 1 60); do
  PACKAGE_PATH="$(adb shell pm path "${PACKAGE}" 2>/dev/null | tr -d '\r' || true)"
  MAIN_RESOLUTION="$(
    adb shell cmd package resolve-activity --brief \
      -a android.intent.action.MAIN \
      -c android.intent.category.LAUNCHER \
      "${PACKAGE}" 2>/dev/null | tr -d '\r' || true
  )"
  if printf '%s\n' "${PACKAGE_PATH}" | grep -q '^package:' \
      && printf '%s\n' "${MAIN_RESOLUTION}" | grep -q "${PACKAGE}"; then
    PACKAGE_READY=true
    break
  fi
  sleep 2
done

test "${PACKAGE_READY}" = "true"
adb shell input keyevent 82 || true

adb shell am start -W -n "${MAIN_ACTIVITY}"
test -n "$(adb shell pidof "${PACKAGE}" | tr -d '\r')"

adb shell am start -W -n "${PROBE_ACTIVITY}"
sleep 2

POST_REBOOT="$(adb shell run-as "${PACKAGE}" cat files/ntd97-lifecycle-probe.txt | tr -d '\r')"
printf '%s\n' "${POST_REBOOT}"

for REQUIRED in \
  build_attestation=ok \
  native_host=ok \
  avatar=ok \
  audio_bridge=ok \
  notification_channels=ok
do
  printf '%s\n' "${POST_REBOOT}" | grep -Fxq "${REQUIRED}"
done

printf '%s\n' "${REAL_MODEL_PROBE}" | grep -Fxq 'web_status=200'
printf '%s\n' "${REAL_MODEL_PROBE}" | grep -Eq '^web_bytes=[1-9][0-9]*

adb shell run-as "${PACKAGE}" rm -f \
  files/ntd97-device-evidence.nde97 \
  files/ntd97-device-evidence.txt || true
adb shell am start -W -n "${PHYSICAL_ACTIVITY}" >/dev/null

for ATTEMPT in $(seq 1 30); do
  if adb shell run-as "${PACKAGE}" test -f files/ntd97-device-evidence.txt; then
    break
  fi
  sleep 1
done

PHYSICAL_REJECTION="$(adb shell run-as "${PACKAGE}" cat files/ntd97-device-evidence.txt | tr -d '\r')"
printf '%s\n' "${PHYSICAL_REJECTION}"
printf '%s\n' "${PHYSICAL_REJECTION}" | grep -Fxq "status=emulator-rejected"

if adb shell run-as "${PACKAGE}" test -f files/ntd97-device-evidence.nde97; then
  echo "emulator unexpectedly produced PhysicalDevice evidence" >&2
  exit 1
fi

adb reboot
adb wait-for-device
for ATTEMPT in $(seq 1 90); do
  if [ "$(adb shell getprop sys.boot_completed | tr -d '\r')" = "1" ]; then
    break
  fi
  sleep 2
done

test "$(adb shell getprop sys.boot_completed | tr -d '\r')" = "1"

PACKAGE_READY=false
for ATTEMPT in $(seq 1 60); do
  PACKAGE_PATH="$(adb shell pm path "${PACKAGE}" 2>/dev/null | tr -d '\r' || true)"
  MAIN_RESOLUTION="$(
    adb shell cmd package resolve-activity --brief \
      -a android.intent.action.MAIN \
      -c android.intent.category.LAUNCHER \
      "${PACKAGE}" 2>/dev/null | tr -d '\r' || true
  )"
  if printf '%s\n' "${PACKAGE_PATH}" | grep -q '^package:' \
      && printf '%s\n' "${MAIN_RESOLUTION}" | grep -q "${PACKAGE}"; then
    PACKAGE_READY=true
    break
  fi
  sleep 2
done

test "${PACKAGE_READY}" = "true"
adb shell input keyevent 82 || true

adb shell am start -W -n "${MAIN_ACTIVITY}"
test -n "$(adb shell pidof "${PACKAGE}" | tr -d '\r')"

adb shell am start -W -n "${PROBE_ACTIVITY}"
sleep 2

POST_REBOOT="$(adb shell run-as "${PACKAGE}" cat files/ntd97-lifecycle-probe.txt | tr -d '\r')"
printf '%s\n' "${POST_REBOOT}"

for REQUIRED in \
  build_attestation=ok \
  native_host=ok \
  avatar=ok \
  audio_bridge=ok \
  notification_channels=ok
do
  printf '%s\n' "${POST_REBOOT}" | grep -Fxq "${REQUIRED}"
done

printf '%s\n' "${REAL_MODEL_PROBE}" | grep -Eq '^web_sha256=[0-9a-f]{64}

adb shell run-as "${PACKAGE}" rm -f \
  files/ntd97-device-evidence.nde97 \
  files/ntd97-device-evidence.txt || true
adb shell am start -W -n "${PHYSICAL_ACTIVITY}" >/dev/null

for ATTEMPT in $(seq 1 30); do
  if adb shell run-as "${PACKAGE}" test -f files/ntd97-device-evidence.txt; then
    break
  fi
  sleep 1
done

PHYSICAL_REJECTION="$(adb shell run-as "${PACKAGE}" cat files/ntd97-device-evidence.txt | tr -d '\r')"
printf '%s\n' "${PHYSICAL_REJECTION}"
printf '%s\n' "${PHYSICAL_REJECTION}" | grep -Fxq "status=emulator-rejected"

if adb shell run-as "${PACKAGE}" test -f files/ntd97-device-evidence.nde97; then
  echo "emulator unexpectedly produced PhysicalDevice evidence" >&2
  exit 1
fi

adb reboot
adb wait-for-device
for ATTEMPT in $(seq 1 90); do
  if [ "$(adb shell getprop sys.boot_completed | tr -d '\r')" = "1" ]; then
    break
  fi
  sleep 2
done

test "$(adb shell getprop sys.boot_completed | tr -d '\r')" = "1"

PACKAGE_READY=false
for ATTEMPT in $(seq 1 60); do
  PACKAGE_PATH="$(adb shell pm path "${PACKAGE}" 2>/dev/null | tr -d '\r' || true)"
  MAIN_RESOLUTION="$(
    adb shell cmd package resolve-activity --brief \
      -a android.intent.action.MAIN \
      -c android.intent.category.LAUNCHER \
      "${PACKAGE}" 2>/dev/null | tr -d '\r' || true
  )"
  if printf '%s\n' "${PACKAGE_PATH}" | grep -q '^package:' \
      && printf '%s\n' "${MAIN_RESOLUTION}" | grep -q "${PACKAGE}"; then
    PACKAGE_READY=true
    break
  fi
  sleep 2
done

test "${PACKAGE_READY}" = "true"
adb shell input keyevent 82 || true

adb shell am start -W -n "${MAIN_ACTIVITY}"
test -n "$(adb shell pidof "${PACKAGE}" | tr -d '\r')"

adb shell am start -W -n "${PROBE_ACTIVITY}"
sleep 2

POST_REBOOT="$(adb shell run-as "${PACKAGE}" cat files/ntd97-lifecycle-probe.txt | tr -d '\r')"
printf '%s\n' "${POST_REBOOT}"

for REQUIRED in \
  build_attestation=ok \
  native_host=ok \
  avatar=ok \
  audio_bridge=ok \
  notification_channels=ok
do
  printf '%s\n' "${POST_REBOOT}" | grep -Fxq "${REQUIRED}"
done


adb shell run-as "${PACKAGE}" rm -f \
  files/ntd97-device-evidence.nde97 \
  files/ntd97-device-evidence.txt || true
adb shell am start -W -n "${PHYSICAL_ACTIVITY}" >/dev/null

for ATTEMPT in $(seq 1 30); do
  if adb shell run-as "${PACKAGE}" test -f files/ntd97-device-evidence.txt; then
    break
  fi
  sleep 1
done

PHYSICAL_REJECTION="$(adb shell run-as "${PACKAGE}" cat files/ntd97-device-evidence.txt | tr -d '\r')"
printf '%s\n' "${PHYSICAL_REJECTION}"
printf '%s\n' "${PHYSICAL_REJECTION}" | grep -Fxq "status=emulator-rejected"

if adb shell run-as "${PACKAGE}" test -f files/ntd97-device-evidence.nde97; then
  echo "emulator unexpectedly produced PhysicalDevice evidence" >&2
  exit 1
fi

adb reboot
adb wait-for-device
for ATTEMPT in $(seq 1 90); do
  if [ "$(adb shell getprop sys.boot_completed | tr -d '\r')" = "1" ]; then
    break
  fi
  sleep 2
done

test "$(adb shell getprop sys.boot_completed | tr -d '\r')" = "1"

PACKAGE_READY=false
for ATTEMPT in $(seq 1 60); do
  PACKAGE_PATH="$(adb shell pm path "${PACKAGE}" 2>/dev/null | tr -d '\r' || true)"
  MAIN_RESOLUTION="$(
    adb shell cmd package resolve-activity --brief \
      -a android.intent.action.MAIN \
      -c android.intent.category.LAUNCHER \
      "${PACKAGE}" 2>/dev/null | tr -d '\r' || true
  )"
  if printf '%s\n' "${PACKAGE_PATH}" | grep -q '^package:' \
      && printf '%s\n' "${MAIN_RESOLUTION}" | grep -q "${PACKAGE}"; then
    PACKAGE_READY=true
    break
  fi
  sleep 2
done

test "${PACKAGE_READY}" = "true"
adb shell input keyevent 82 || true

adb shell am start -W -n "${MAIN_ACTIVITY}"
test -n "$(adb shell pidof "${PACKAGE}" | tr -d '\r')"

adb shell am start -W -n "${PROBE_ACTIVITY}"
sleep 2

POST_REBOOT="$(adb shell run-as "${PACKAGE}" cat files/ntd97-lifecycle-probe.txt | tr -d '\r')"
printf '%s\n' "${POST_REBOOT}"

for REQUIRED in \
  build_attestation=ok \
  native_host=ok \
  avatar=ok \
  audio_bridge=ok \
  notification_channels=ok
do
  printf '%s\n' "${POST_REBOOT}" | grep -Fxq "${REQUIRED}"
done
