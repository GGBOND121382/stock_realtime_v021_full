#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Archive saved_data/*_pipeline_out directories, then delete CSV files inside them.

Usage:
  bash scripts/archive_pipeline_out_and_delete_csv.sh --yes

Options:
  --saved-data DIR     saved_data directory; default: saved_data
  --archive-dir DIR    output directory; default: saved_data/pipeline_out_archives/YYYYmmdd_HHMMSS
  --yes                actually delete CSV files after each zip passes `zip -T`
  --dry-run            print actions without writing archives or deleting files
  -h, --help           show this help

Notes:
  - Only immediate child directories matching *_pipeline_out are archived.
  - CSV deletion is recursive inside each *_pipeline_out directory.
  - Existing archives are never overwritten because archive-dir defaults to a timestamped directory.
EOF
}

saved_data="saved_data"
archive_dir=""
confirm_delete=0
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --saved-data)
      saved_data="${2:?missing value for --saved-data}"
      shift 2
      ;;
    --archive-dir)
      archive_dir="${2:?missing value for --archive-dir}"
      shift 2
      ;;
    --yes)
      confirm_delete=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "$saved_data" ]]; then
  echo "saved_data directory not found: $saved_data" >&2
  exit 1
fi

if ! command -v zip >/dev/null 2>&1; then
  echo "zip command not found. Install it first, e.g. sudo apt-get install -y zip" >&2
  exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
if [[ -z "$archive_dir" ]]; then
  archive_dir="$saved_data/pipeline_out_archives/$timestamp"
fi

if [[ "$dry_run" -eq 0 && "$confirm_delete" -ne 1 ]]; then
  echo "Refusing to delete CSV files without --yes. Use --dry-run to preview." >&2
  exit 2
fi

mapfile -d '' pipeline_dirs < <(find "$saved_data" -mindepth 1 -maxdepth 1 -type d -name '*_pipeline_out' -print0 | sort -z)

if [[ "${#pipeline_dirs[@]}" -eq 0 ]]; then
  echo "No *_pipeline_out directories found under $saved_data"
  exit 0
fi

echo "saved_data:  $saved_data"
echo "archive_dir: $archive_dir"
echo "directories: ${#pipeline_dirs[@]}"
echo

if [[ "$dry_run" -eq 0 ]]; then
  mkdir -p "$archive_dir"
  archive_dir_abs="$(cd "$archive_dir" && pwd)"
else
  archive_dir_abs="$archive_dir"
fi

for dir in "${pipeline_dirs[@]}"; do
  name="$(basename "$dir")"
  zip_path="$archive_dir_abs/${name}.zip"
  csv_count="$(find "$dir" -type f -iname '*.csv' | wc -l | tr -d ' ')"

  echo "==> $name"
  echo "    csv files: $csv_count"
  echo "    archive:   $zip_path"

  if [[ "$dry_run" -eq 1 ]]; then
    echo "    dry-run: would zip '$dir' and delete $csv_count CSV files after verification"
    continue
  fi

  if [[ -e "$zip_path" ]]; then
    echo "Archive already exists, refusing to overwrite: $zip_path" >&2
    exit 1
  fi

  (
    cd "$saved_data"
    zip -qr "$zip_path" "$name"
  )

  zip -T "$zip_path" >/dev/null

  if [[ "$csv_count" -gt 0 ]]; then
    find "$dir" -type f -iname '*.csv' -delete
  fi

  remaining_csv_count="$(find "$dir" -type f -iname '*.csv' | wc -l | tr -d ' ')"
  echo "    done: zip verified; remaining CSV files: $remaining_csv_count"
done

echo
echo "All done."
