#!/usr/bin/env bash
#
# pack_data.sh
# -----------------------------------------------------------------------------
# Compress the large evaluation data directories and split them into shards
# that can be committed to GitHub (100 MB per-file limit).
#
# What gets packed is controlled by the INCLUDE_* whitelist below.
# Only files actually needed for evaluation are included:
#   - sandbox_cache/*_cache.json         sandbox tool-call cache (10 tools)
#   - poi2category/all_poi_by_city.json  POI metadata (read during evaluation)
#
# Everything else (*_missed.json, *_embeddings.npz, intermediates, logs)
# is excluded so runtime leftovers are never published.
#
# Outputs:
#   data_archive/<name>/<name>.tar.gz.part.*  shards
#   data_archive/manifest.sha256              sha256 of original files (checked after restore)
#   data_archive/archive.sha256               sha256 of each shard (checked before unpacking)
#
# Usage:
#   bash scripts/pack_data.sh
# -----------------------------------------------------------------------------
set -euo pipefail

# ---- config ----
# Per-shard size. GitHub's hard limit is 100 MB; 95M leaves a safety margin.
CHUNK_SIZE="95M"
# Directories to pack (relative to the repository root)
DIRS=("sandbox_cache" "poi2category")
# Per-directory file whitelist (find -name patterns, space-separated)
INCLUDE_sandbox_cache="*_cache.json"
INCLUDE_poi2category="all_poi_by_city.json"
# Shard output directory
ARCHIVE_DIR="data_archive"

# ---- locate the repository root ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---- sha256 helper (Linux: sha256sum, macOS: shasum -a 256) ----
if command -v sha256sum >/dev/null 2>&1; then
    sha256() { sha256sum "$@"; }
elif command -v shasum >/dev/null 2>&1; then
    sha256() { shasum -a 256 "$@"; }
else
    echo "!! Neither sha256sum nor shasum found; cannot write checksums."
    exit 1
fi

echo "==> Repository root: ${REPO_ROOT}"
echo "==> Shard size:      ${CHUNK_SIZE}"
echo "==> Output dir:      ${ARCHIVE_DIR}/"
echo

MANIFEST="${ARCHIVE_DIR}/manifest.sha256"
ARCHIVE_SUMS="${ARCHIVE_DIR}/archive.sha256"

mkdir -p "${ARCHIVE_DIR}"
: > "${MANIFEST}"
: > "${ARCHIVE_SUMS}"

for name in "${DIRS[@]}"; do
    if [[ ! -d "${name}" ]]; then
        echo "!! Skip: directory does not exist: ${name}"
        continue
    fi

    echo "==> Packing: ${name}/"

    # 1) Collect files matching the whitelist
    patterns_var="INCLUDE_${name}"
    patterns="${!patterns_var:-}"
    if [[ -z "${patterns}" ]]; then
        echo "!! Skip: no whitelist for ${name} (INCLUDE_${name})"
        continue
    fi

    find_args=()
    for pattern in ${patterns}; do
        find_args+=(-name "${pattern}" -o)
    done
    unset 'find_args[${#find_args[@]}-1]'   # drop the trailing -o

    file_list="$(mktemp)"
    find "${name}" -type f \( "${find_args[@]}" \) -print | sort > "${file_list}"

    file_count=$(wc -l < "${file_list}" | tr -d ' ')
    if [[ "${file_count}" -eq 0 ]]; then
        echo "!! Skip: no files in ${name}/ match '${patterns}'"
        rm -f "${file_list}"
        continue
    fi
    echo "    - ${file_count} file(s) matched (whitelist: ${patterns})"
    sed 's/^/        /' "${file_list}"

    # 2) Record sha256 of the original files (checked after restore)
    echo "    - Checksumming original files ..."
    while IFS= read -r f; do
        sha256 "${f}" >> "${MANIFEST}"
    done < "${file_list}"

    # 3) Reset this directory's shard output
    out_dir="${ARCHIVE_DIR}/${name}"
    rm -rf "${out_dir}"
    mkdir -p "${out_dir}"

    # 4) Pack + compress + split
    echo "    - Compressing and splitting ..."
    prefix="${out_dir}/${name}.tar.gz.part."
    tar -cf - -T "${file_list}" | gzip -c | split -b "${CHUNK_SIZE}" - "${prefix}"
    rm -f "${file_list}"

    # 5) Record shard checksums (paths relative to ARCHIVE_DIR for restore)
    echo "    - Checksumming shards ..."
    (
        cd "${ARCHIVE_DIR}"
        find "${name}" -type f -name "*.tar.gz.part.*" -print | sort | while IFS= read -r part; do
            sha256 "${part}"
        done
    ) >> "${ARCHIVE_SUMS}"

    # 6) Show results
    echo "    - Shards written:"
    ls -lh "${out_dir}" | awk 'NR>1 {print "        " $5 "  " $9}'
    echo
done

echo "==> Checking that every shard is under 100 MB ..."
oversized=$(find "${ARCHIVE_DIR}" -type f -size +100M || true)
if [[ -n "${oversized}" ]]; then
    echo "!! Warning: the following shards still exceed 100 MB. Lower CHUNK_SIZE and rerun:"
    echo "${oversized}"
    exit 1
fi
echo "    All shards are under 100 MB."
echo
echo "==> Done. Shards are in ${ARCHIVE_DIR}/, checksums:"
echo "      - ${MANIFEST}"
echo "      - ${ARCHIVE_SUMS}"
echo
echo "You can now commit ${ARCHIVE_DIR}/."
echo "To restore: bash scripts/restore_data.sh"
