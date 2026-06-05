#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="$ROOT_DIR/generated"
SCHEMA="$ROOT_DIR/LongKSoftmax.json"
CANN_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}
COMPUTE_UNIT=${ASCEND_COMPUTE_UNIT:-ai_core-ascend910}
MSOPGEN_SCRIPT="$CANN_PATH/bin/msopgen"
if [[ ! -f "$MSOPGEN_SCRIPT" ]]; then
  MSOPGEN_SCRIPT="$CANN_PATH/python/site-packages/bin/msopgen"
fi
if [[ ! -f "$MSOPGEN_SCRIPT" ]]; then
  echo "msopgen script not found under $CANN_PATH." >&2
  exit 1
fi
chmod go-w "$SCHEMA" "$ROOT_DIR" "$ROOT_DIR/.." "$ROOT_DIR/../.."
rm -rf "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR"
chmod go-w "$PROJECT_DIR"
python "$MSOPGEN_SCRIPT" gen -i "$SCHEMA" -f pytorch -c "$COMPUTE_UNIT" -lan cpp -out "$PROJECT_DIR"
python - <<PY
from pathlib import Path
path = Path("$PROJECT_DIR") / "CMakePresets.json"
text = path.read_text()
text = text.replace('"value": "customize"', '"value": "ltx2_ascend"')
path.write_text(text)
PY
