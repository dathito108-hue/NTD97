#!/usr/bin/env bash
set -euo pipefail

APK="${1:-platform/android/app/build/outputs/apk/debug/app-debug.apk}"
OUTPUT_DIR="${2:-physical-evidence}"
PACKAGE="ai.ntd97.mobile"
ACTIVITY="${PACKAGE}/.NtdPhysicalEvidenceActivity"

test -f "${APK}"
command -v adb >/dev/null

if [[ "$(adb shell getprop ro.kernel.qemu 2>/dev/null | tr -d '\r')" == "1" ]]; then
  echo "Refusing to collect PhysicalDevice evidence from an emulator." >&2
  exit 3
fi

mkdir -p "${OUTPUT_DIR}"
adb install -r "${APK}" >/dev/null

adb shell run-as "${PACKAGE}" rm -f \
  files/ntd97-device-evidence.nde97 \
  files/ntd97-device-evidence.txt || true

adb shell am start -W -n "${ACTIVITY}" >/dev/null

for _ in $(seq 1 180); do
  if adb shell run-as "${PACKAGE}" test -f files/ntd97-device-evidence.txt; then
    break
  fi
  sleep 1
done

adb shell run-as "${PACKAGE}" test -f files/ntd97-device-evidence.txt

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SUMMARY="${OUTPUT_DIR}/ntd97-${STAMP}.txt"
EVIDENCE="${OUTPUT_DIR}/ntd97-${STAMP}.nde97"

adb exec-out run-as "${PACKAGE}" cat files/ntd97-device-evidence.txt > "${SUMMARY}"
cat "${SUMMARY}"

if grep -q '^status=emulator-rejected$' "${SUMMARY}"; then
  echo "Collector rejected this device as an emulator." >&2
  exit 3
fi

if ! grep -q '^status=ok$' "${SUMMARY}"; then
  echo "Physical evidence collector did not complete successfully." >&2
  exit 2
fi

if adb shell run-as "${PACKAGE}" test -f files/ntd97-device-evidence.nde97; then
  adb exec-out run-as "${PACKAGE}" cat files/ntd97-device-evidence.nde97 > "${EVIDENCE}"
  test -s "${EVIDENCE}"
  echo "Evidence: ${EVIDENCE}"
  echo "Summary:  ${SUMMARY}"
else
  echo "No binary evidence was produced. Check the summary above." >&2
  exit 2
fi
