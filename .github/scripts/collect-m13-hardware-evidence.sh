#!/usr/bin/env bash
set -euo pipefail

APK="${1:-platform/android/app/build/outputs/apk/debug/app-debug.apk}"
OUTPUT_DIR="${2:-m13-hardware-evidence}"
EXPECTED_REVISION="${3:-}"
PACKAGE="ai.ntd97.mobile"
ACTIVITY="${PACKAGE}/.NtdM13HardwareEvidenceActivity"
SUMMARY_REL="files/ntd97-m13-hardware-evidence.txt"
EVIDENCE_REL="files/ntd97-m13-hardware-evidence.m13e97"

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
}

b64() {
  printf '%s' "$1" | base64 | tr -d '\r\n'
}

read_summary() {
  adb exec-out run-as "${PACKAGE}" cat "${SUMMARY_REL}" 2>/dev/null || true
}

wait_for_status() {
  local expected="$1"
  local attempts="${2:-360}"
  local summary=""
  for _ in $(seq 1 "${attempts}"); do
    summary="$(read_summary)"
    if grep -q '^status=failed$' <<<"${summary}"; then
      printf '%s\n' "${summary}" >&2
      return 2
    fi
    if grep -q "^status=${expected}$" <<<"${summary}"; then
      printf '%s' "${summary}"
      return 0
    fi
    sleep 1
  done
  printf '%s\n' "${summary}" >&2
  echo "Timed out waiting for status=${expected}" >&2
  return 2
}

for name in   NTD97_M13_SEARCH_QUERY   NTD97_M13_BROWSER_URL   NTD97_M13_STORAGE_PATH   NTD97_M13_UPLOAD_URL   NTD97_M13_ACCESSIBILITY_PACKAGE   NTD97_M13_ACCESSIBILITY_VIEW_ID   NTD97_M13_PC_ALIAS   NTD97_M13_PC_SURFACE   NTD97_M13_PC_PATH
do
  require_env "${name}"
done

test -f "${APK}"
command -v adb >/dev/null
command -v base64 >/dev/null

adb get-state >/dev/null
if [[ "$(adb shell getprop ro.kernel.qemu 2>/dev/null | tr -d '\r')" == "1" ]]; then
  echo "Refusing to collect M13 hardware evidence from an emulator." >&2
  exit 3
fi

if [[ -z "${EXPECTED_REVISION}" ]]; then
  EXPECTED_REVISION="$(git rev-parse HEAD 2>/dev/null || true)"
fi
if [[ ! "${EXPECTED_REVISION}" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "Expected canonical revision must be a 40-character Git SHA." >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
adb install -r "${APK}" >/dev/null

echo "Checking user-controlled AccessibilityService enablement..."
ENABLED_SERVICES="$(adb shell settings get secure enabled_accessibility_services 2>/dev/null | tr -d '\r')"
if [[ "${ENABLED_SERVICES}" != *"ai.ntd97.mobile/.NtdAccessibilityService"* ]]; then
  echo "NTD97 AccessibilityService is not enabled. Enable it manually in Android Accessibility Settings before this run." >&2
  exit 2
fi

echo "Phase A: running production mixed-surface actions..."
adb shell am force-stop "${PACKAGE}" >/dev/null || true
adb shell run-as "${PACKAGE}" rm -f "${SUMMARY_REL}" "${EVIDENCE_REL}" >/dev/null 2>&1 || true

adb shell am start -W -n "${ACTIVITY}"   --es phase prepare   --ez approve_external true   --es search_query_b64 "$(b64 "${NTD97_M13_SEARCH_QUERY}")"   --es browser_url_b64 "$(b64 "${NTD97_M13_BROWSER_URL}")"   --es storage_path_b64 "$(b64 "${NTD97_M13_STORAGE_PATH}")"   --es upload_url_b64 "$(b64 "${NTD97_M13_UPLOAD_URL}")"   --es accessibility_package_b64 "$(b64 "${NTD97_M13_ACCESSIBILITY_PACKAGE}")"   --es accessibility_view_id_b64 "$(b64 "${NTD97_M13_ACCESSIBILITY_VIEW_ID}")"   --es pc_alias_b64 "$(b64 "${NTD97_M13_PC_ALIAS}")"   --es pc_surface_b64 "$(b64 "${NTD97_M13_PC_SURFACE}")"   --es pc_path_b64 "$(b64 "${NTD97_M13_PC_PATH}")" >/dev/null

PHASE_A_SUMMARY="$(wait_for_status awaiting-process-death 600)"
printf '%s\n' "${PHASE_A_SUMMARY}"

BUILD_REVISION="$(awk -F= '$1=="build_revision"{print $2; exit}' <<<"${PHASE_A_SUMMARY}")"
if [[ "${BUILD_REVISION}" != "${EXPECTED_REVISION}" ]]; then
  echo "APK revision mismatch: expected ${EXPECTED_REVISION}, got ${BUILD_REVISION}" >&2
  exit 2
fi

echo "Forcing a real Android process death before upload resume..."
adb shell am force-stop "${PACKAGE}" >/dev/null
for _ in $(seq 1 30); do
  if [[ -z "$(adb shell pidof "${PACKAGE}" 2>/dev/null | tr -d '\r')" ]]; then
    break
  fi
  sleep 1
done
if [[ -n "$(adb shell pidof "${PACKAGE}" 2>/dev/null | tr -d '\r')" ]]; then
  echo "App process did not terminate after force-stop." >&2
  exit 2
fi

echo "Phase B: restoring sovereign ActionFabric and requiring reconfirm..."
adb shell am start -W -n "${ACTIVITY}"   --es phase resume   --ez approve_external true >/dev/null

PHASE_B_SUMMARY="$(wait_for_status ok 600)"
printf '%s\n' "${PHASE_B_SUMMARY}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SUMMARY="${OUTPUT_DIR}/ntd97-m13-${STAMP}.txt"
EVIDENCE="${OUTPUT_DIR}/ntd97-m13-${STAMP}.m13e97"

adb exec-out run-as "${PACKAGE}" cat "${SUMMARY_REL}" > "${SUMMARY}"
adb exec-out run-as "${PACKAGE}" cat "${EVIDENCE_REL}" > "${EVIDENCE}"
test -s "${SUMMARY}"
test -s "${EVIDENCE}"

grep -q '^status=ok$' "${SUMMARY}"
grep -q "^build_revision=${EXPECTED_REVISION}$" "${SUMMARY}"
grep -q '^process_death_reconfirmed=true$' "${SUMMARY}"
grep -q '^sovereignty_audit_passed=true$' "${SUMMARY}"

if command -v cargo >/dev/null; then
  cargo run -q -p ntd-validation --bin m13e97-gate --     "${EXPECTED_REVISION}" "${EVIDENCE}"
else
  echo "cargo not found; run m13e97-gate manually before accepting this evidence." >&2
  exit 2
fi

echo "M13 hardware evidence: ${EVIDENCE}"
echo "M13 summary:           ${SUMMARY}"
