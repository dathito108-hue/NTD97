#!/usr/bin/env bash
set -euo pipefail

APK="platform/android/app/build/outputs/apk/debug/app-debug.apk"
PACKAGE="ai.ntd97.mobile"
PROBE_ACTIVITY="${PACKAGE}/.NtdLifecycleProbeActivity"
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
  native_host=ok \
  avatar=ok \
  audio_bridge=ok \
  notification_channels=ok \
  foreground_request=ok \
  overlay_request=ok \
  persistent_job=ok \
  reboot_marker=ok

do
  printf '%s\n' "${PROBE}" | grep -q "^${REQUIRED}$"
done

adb shell dumpsys activity services "${PACKAGE}" | grep -q "NtdOverlayService"

adb reboot
adb wait-for-device
for ATTEMPT in $(seq 1 90); do
  if [ "$(adb shell getprop sys.boot_completed | tr -d '\r')" = "1" ]; then
    break
  fi
  sleep 2
done

test "$(adb shell getprop sys.boot_completed | tr -d '\r')" = "1"
adb shell input keyevent 82 || true

adb shell am start -W -n "${MAIN_ACTIVITY}"
test -n "$(adb shell pidof "${PACKAGE}" | tr -d '\r')"

adb shell am start -W -n "${PROBE_ACTIVITY}"
sleep 2

POST_REBOOT="$(adb shell run-as "${PACKAGE}" cat files/ntd97-lifecycle-probe.txt | tr -d '\r')"
printf '%s\n' "${POST_REBOOT}"

for REQUIRED in \
  native_host=ok \
  avatar=ok \
  audio_bridge=ok \
  notification_channels=ok

do
  printf '%s\n' "${POST_REBOOT}" | grep -q "^${REQUIRED}$"
done
