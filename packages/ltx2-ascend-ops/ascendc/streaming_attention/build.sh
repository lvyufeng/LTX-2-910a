#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="$ROOT_DIR/generated"
CANN_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}

_append_env_path() {
  local name=$1
  local value=$2
  if [[ -z "$value" || ! -d "$value" ]]; then
    return
  fi
  local current=${!name:-}
  case ":$current:" in
    *":$value:"*) ;;
    *) export "$name"="${current:+$current:}$value" ;;
  esac
}

_configure_host_cxx_includes() {
  local gcc_include version triple ordered current extra flags path flag
  gcc_include=$(g++ -print-file-name=include 2>/dev/null || true)
  if [[ -n "$gcc_include" && -d "$gcc_include" ]]; then
    version=$(basename "$gcc_include")
    triple=$(basename "$(dirname "$gcc_include")")
  else
    version=11
    triple=aarch64-linux-gnu
  fi

  ordered=""
  for path in \
    "$gcc_include" \
    "/usr/include/$triple" \
    "/usr/include/$triple/c++/$version" \
    "/usr/include/c++/$version" \
    "/usr/lib/gcc/aarch64-linux-gnu/11/include" \
    "/usr/include/aarch64-linux-gnu" \
    "/usr/include/aarch64-linux-gnu/c++/11" \
    "/usr/include/c++/11" \
    "/usr/include"; do
    [[ -n "$path" && -d "$path" ]] || continue
    case ":$ordered:" in
      *":$path:"*) ;;
      *) ordered="${ordered:+$ordered:}$path" ;;
    esac
  done

  current=${CPLUS_INCLUDE_PATH:-}
  extra=""
  IFS=':' read -r -a include_parts <<< "$current"
  for path in "${include_parts[@]}"; do
    [[ -n "$path" && -d "$path" ]] || continue
    case ":$ordered:$extra:" in
      *":$path:"*) ;;
      *) extra="${extra:+$extra:}$path" ;;
    esac
  done
  export CPLUS_INCLUDE_PATH="$ordered${extra:+:$extra}"

  flags=${CXXFLAGS:-}
  IFS=':' read -r -a include_parts <<< "$CPLUS_INCLUDE_PATH"
  for path in "${include_parts[@]}"; do
    [[ -z "$path" ]] && continue
    flag="-I$path"
    case " $flags " in
      *" $flag "*) ;;
      *) flags="${flags:+$flags }$flag" ;;
    esac
  done
  export CXXFLAGS="$flags"
}

if [[ ! -f "$PROJECT_DIR/build.sh" ]]; then
  echo "Generated AscendC project not found at $PROJECT_DIR." >&2
  echo "Run $ROOT_DIR/generate.sh first, then implement the generated host/kernel TODOs." >&2
  exit 1
fi

if [[ -f "$CANN_PATH/set_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "$CANN_PATH/set_env.sh"
fi
_configure_host_cxx_includes

cd "$PROJECT_DIR"
bash build.sh
