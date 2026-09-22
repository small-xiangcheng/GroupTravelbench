#!/usr/bin/env bash
#
# restore_data.sh
# -----------------------------------------------------------------------------
# Reassemble the shards under data_archive/, unpack the evaluation data
# directories, and verify every file with sha256 so the restored data matches
# the packed originals exactly.
#
# Restored directories (must stay in sync with pack_data.sh):
#   - sandbox_cache/   sandbox tool-call cache (read during inference)
#   - poi2category/    POI metadata (read during evaluation)
#
# Usage:
#   bash scripts/restore_data.sh
# -----------------------------------------------------------------------------
set -euo pipefail

# ---- config (must match pack_data.sh) ----
DIRS=("sandbox_cache" "poi2category")
ARCHIVE_DIR="data_archive"

# ---- locate the repository root ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---- sha256 helper (Linux: sha256sum, macOS: shasum -a 256) ----
if command -v sha256sum >/dev/null 2>&1; then
    sha256_check() { sha256sum -c --quiet "$@"; }
elif command -v shasum >/dev/null 2>&1; then
    sha256_check() { shasum -a 256 -c --quiet "$@"; }
else
    echo "!! Neither sha256sum nor shasum found; cannot verify data integrity."
    exit 1
fi

echo "==> Repository root: ${REPO_ROOT}"
echo "==> Shard directory: ${ARCHIVE_DIR}/"
echo

MANIFEST="${ARCHIVE_DIR}/manifest.sha256"
ARCHIVE_SUMS="${ARCHIVE_DIR}/archive.sha256"

if [[ ! -f "${MANIFEST}" ]]; then
    echo "!! Checksum manifest not found: ${MANIFEST}. Cannot restore."
    exit 1
fi

# 1) Verify the shards themselves
if [[ -f "${ARCHIVE_SUMS}" ]]; then
    echo "==> Verifying shard integrity ..."
    ( cd "${ARCHIVE_DIR}" && sha256_check "$(basename "${ARCHIVE_SUMS}")" )
    echo "    Shard integrity check passed."
    echo
fi

# 2) Reassemble and unpack each directory
for name in "${DIRS[@]}"; do
    out_dir="${ARCHIVE_DIR}/${name}"
    if [[ ! -d "${out_dir}" ]]; then
        echo "!! Skip: shard directory not found: ${out_dir}"
        continue
    fi

    echo "==> Restoring: ${name}/"

    # If the destination already exists, rename it so nothing is overwritten
    if [[ -e "${name}" ]]; then
        backup="${name}.bak.$(date +%Y%m%d%H%M%S)"
        echo "    - ${name}/ already exists; renaming to ${backup}/"
        mv "${name}" "${backup}"
    fi

    echo "    - Concatenating shards and unpacking ..."
    # Shard names look like <name>.tar.gz.part.aa/ab/ac...; lexicographic
    # order matches the original split order.
    cat "${out_dir}/${name}.tar.gz.part."* | gzip -dc | tar -xf - -C "${REPO_ROOT}"
    echo "    - Unpack complete."
    echo
done

# 3) Verify every file matches the packed original
echo "==> Verifying sha256 of restored files ..."
if sha256_check "${MANIFEST}"; then
    echo
    echo "==> All checks passed. Data restore is complete."
    echo "    You can now run: bash scripts/run_pipeline.sh --bg"
else
    echo
    echo "!! Check failed: restored data does not match the manifest. See FAILED items above."
    exit 1
fi
