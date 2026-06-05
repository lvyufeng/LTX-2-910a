#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CANN_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}

if [[ -f "$CANN_PATH/set_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "$CANN_PATH/set_env.sh"
fi

cd "$ROOT_DIR"
python csrc/setup_streaming_attention.py build_ext --inplace

cat <<EOF
Built the optional torch.ops binding in:
  $ROOT_DIR/src/ltx2_ascend_ops/

Before runtime validation, install/export the generated custom OPP and enable native dispatch, e.g.:
  export ASCEND_CUSTOM_OPP_PATH="\$ASCEND_OPP_PATH/vendors/ltx2_ascend:\${ASCEND_CUSTOM_OPP_PATH:-}"
  export LD_LIBRARY_PATH="\$ASCEND_OPP_PATH/vendors/ltx2_ascend/op_api/lib:\$LD_LIBRARY_PATH"
  export LTX2_ASCEND_STREAMING_ATTN_ENABLE_NATIVE=1
  export LTX2_ASCEND_ATTENTION=streaming
EOF
