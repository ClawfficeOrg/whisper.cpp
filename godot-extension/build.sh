#!/usr/bin/env bash
# build.sh — Build the whisper_cpp_gdext GDExtension and copy output
# into the careless-whisper-godot Godot project.
#
# Usage:
#   ./build.sh [--release] [--godot-project /path/to/godot/project]
#
# Environment:
#   GODOT_PROJECT  Path to the Godot project root (default: ../../careless-whisper-godot)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="debug"
GODOT_PROJECT="${GODOT_PROJECT:-${SCRIPT_DIR}/../../careless-whisper-godot}"

for arg in "$@"; do
    case "$arg" in
        --release) PROFILE="release" ;;
        --godot-project=*) GODOT_PROJECT="${arg#*=}" ;;
    esac
done

echo "==> Building whisper_cpp_gdext (profile: ${PROFILE})"
cd "${SCRIPT_DIR}"
cargo build $([ "$PROFILE" = "release" ] && echo "--release" || true)

TARGET_DIR="${SCRIPT_DIR}/target/${PROFILE}"
OUT_DIR="${GODOT_PROJECT}/addons/whisper_cpp/bin"
mkdir -p "${OUT_DIR}"

echo "==> Copying binaries to ${OUT_DIR}"
if [ -f "${TARGET_DIR}/libwhisper_cpp_gdext.so" ]; then
    cp "${TARGET_DIR}/libwhisper_cpp_gdext.so" \
       "${OUT_DIR}/libwhisper_cpp_gdext.linux.${PROFILE}.x86_64.so"
    echo "    -> linux .so copied"
fi
if [ -f "${TARGET_DIR}/whisper_cpp_gdext.dll" ]; then
    cp "${TARGET_DIR}/whisper_cpp_gdext.dll" \
       "${OUT_DIR}/whisper_cpp_gdext.windows.${PROFILE}.x86_64.dll"
    echo "    -> windows .dll copied"
fi
if [ -f "${TARGET_DIR}/libwhisper_cpp_gdext.dylib" ]; then
    cp "${TARGET_DIR}/libwhisper_cpp_gdext.dylib" \
       "${OUT_DIR}/libwhisper_cpp_gdext.macos.${PROFILE}.framework"
    echo "    -> macos .dylib copied"
fi

# Copy .gdextension manifest into addons directory
cp "${SCRIPT_DIR}/whisper_cpp.gdextension" "${GODOT_PROJECT}/addons/whisper_cpp/whisper_cpp.gdextension"
echo "==> Done. Run Godot and the WhisperCpp node should be available."
