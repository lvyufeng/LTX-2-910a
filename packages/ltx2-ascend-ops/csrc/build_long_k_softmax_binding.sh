#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CANN_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}

if [[ -f "$CANN_PATH/set_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "$CANN_PATH/set_env.sh"
fi

cd "$ROOT_DIR"
python csrc/setup_long_k_softmax.py build_ext --inplace

cat <<EOF
Built the optional long-K softmax torch.ops binding in:
  $ROOT_DIR/src/ltx2_ascend_ops/

Before runtime validation, install/export the generated custom OPP and enable native dispatch, e.g.:
  export ASCEND_CUSTOM_OPP_PATH="\$ASCEND_OPP_PATH/vendors/ltx2_ascend:\${ASCEND_CUSTOM_OPP_PATH:-}"
  export LD_LIBRARY_PATH="\$ASCEND_OPP_PATH/vendors/ltx2_ascend/op_api/lib:\$LD_LIBRARY_PATH"
  export LTX2_ASCEND_LONGK_SOFTMAX_NATIVE=1

For local bring-up without installing the op_api lib into ASCEND_OPP_PATH, also set:
  export LTX2_ASCEND_LONGK_SOFTMAX_LIB="$ROOT_DIR/ascendc/long_k_softmax/generated/build_out/op_host/libcust_opapi.so"
EOF
