#!/usr/bin/env bash
set -euo pipefail

APK="platform/android/app/build/outputs/apk/debug/app-debug.apk"
PACKAGE="ai.ntd97.mobile"
ACTIVITY="${PACKAGE}/.NtdRealModelProbeActivity"
EXPORT_ROOT="${1:-native-export}"
STAGE_ROOT="/data/local/tmp/ntd97-real-model"

test -f "${APK}"
test -f "${EXPORT_ROOT}/stories260K.ncc97"
test -d "${EXPORT_ROOT}/shards"
test "$(find "${EXPORT_ROOT}/shards" -maxdepth 1 -type f -name '*.ntp97' | wc -l)" -eq 52

adb install -r "${APK}" >/dev/null
adb shell rm -rf "${STAGE_ROOT}"
adb push "${EXPORT_ROOT}" "${STAGE_ROOT}" >/dev/null

adb shell run-as "${PACKAGE}" sh -c \
  "'rm -rf files/ntd97-real-model files/ntd97-real-model-probe.txt &&
    mkdir -p files/ntd97-real-model/shards &&
    cp ${STAGE_ROOT}/stories260K.ncc97 files/ntd97-real-model/stories260K.ncc97 &&
    cp ${STAGE_ROOT}/shards/*.ntp97 files/ntd97-real-model/shards/'"

adb shell am start -W -n "${ACTIVITY}" >/dev/null

for ATTEMPT in $(seq 1 240); do
  if adb shell run-as "${PACKAGE}" test -f files/ntd97-real-model-probe.txt; then
    break
  fi
  sleep 1
done

adb shell run-as "${PACKAGE}" test -f files/ntd97-real-model-probe.txt
PROBE="$(adb shell run-as "${PACKAGE}" cat files/ntd97-real-model-probe.txt | tr -d '\r')"
printf '%s\n' "${PROBE}"

printf '%s\n' "${PROBE}" | grep -Fxq "real_model_package=ok"
printf '%s\n' "${PROBE}" | grep -Fxq "real_model_generation=ok"
printf '%s\n' "${PROBE}" | grep -Fxq "generated_tokens=128"
printf '%s\n' "${PROBE}" | grep -Fxq "generated_bytes=322"
printf '%s\n' "${PROBE}" | grep -Fxq \
  "generated_sha256=594a911ebb2ecfeb608919bf157887e82d0090507fa187d45b2b7e23e5e8f583"

if adb shell run-as "${PACKAGE}" ls files/ntd97-real-model 2>/dev/null | grep -Eq 'gguf|\.bin$|run\.c'; then
  echo "source-runtime artifact leaked into Android model sandbox" >&2
  exit 1
fi

echo "android_real_native_model=PASS"
