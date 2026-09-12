"""
ComfyUI-FLASepformer - FLASepformer Speech Separation Node
Author: FY
Description: Speech separation node using FLASepformer model (DAMO model)
"""

import os
import sys
import subprocess
import tempfile
import numpy as np
import soundfile as sf
import onnxruntime as rt
import folder_paths

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
temp_dir = os.path.join(current_dir, "temp")
os.makedirs(temp_dir, exist_ok=True)

# Model path (fixed local path)
MODEL_DIR = os.path.join(
    folder_paths.models_dir,
    "diffusers",
    "speech_flatsepreformer_separation_temporal_8k_base_libri2mix100"
)
ONNX_MODEL_PATH = os.path.join(MODEL_DIR, "onnx_model.onnx")

# Model identifiers
MODELSCOPE_ID = "iic/speech_flatsepreformer_separation_temporal_8k_base_libri2mix100"
HUGGINGFACE_ID = "damo/speech_flatsepreformer_separation_temporal_8k_base_libri2mix100"

# CLI tool paths (derived from Python executable location)
_SCRIPTS_DIR = os.path.join(os.path.dirname(sys.executable), "Scripts")
MODELSCOPE_CLI = os.path.join(_SCRIPTS_DIR, "modelscope.exe")
HUGGINGFACE_CLI = os.path.join(_SCRIPTS_DIR, "huggingface-cli.exe")


def _run_cli(cmd, description):
    """Run a CLI command and return (success, output)."""
    print(f"[FY_FLASepformer] {description}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes max
        )
        if result.returncode == 0:
            print(f"[FY_FLASepformer] {description} — success.")
            return True
        else:
            err = (result.stderr or result.stdout or "").strip()
            print(f"[FY_FLASepformer] {description} — failed (exit {result.returncode}): {err[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"[FY_FLASepformer] {description} — timed out.")
        return False
    except Exception as e:
        print(f"[FY_FLASepformer] {description} — error: {e}")
        return False


def ensure_model_downloaded():
    """Ensure the ONNX model exists locally; auto-download via CLI if missing."""
    if os.path.exists(ONNX_MODEL_PATH):
        return

    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"[FY_FLASepformer] Model not found at {ONNX_MODEL_PATH}, starting download...")

    # 1. Try ModelScope CLI
    if os.path.exists(MODELSCOPE_CLI):
        if _run_cli(
            [MODELSCOPE_CLI, "download", "--model", MODELSCOPE_ID, "--local_dir", MODEL_DIR],
            f"Downloading from ModelScope ({MODELSCOPE_ID})"
        ):
            if os.path.exists(ONNX_MODEL_PATH):
                return

    # 2. Fallback: HuggingFace CLI
    if os.path.exists(HUGGINGFACE_CLI):
        if _run_cli(
            [HUGGINGFACE_CLI, "download", "--local-dir", MODEL_DIR, HUGGINGFACE_ID],
            f"Downloading from HuggingFace ({HUGGINGFACE_ID})"
        ):
            if os.path.exists(ONNX_MODEL_PATH):
                return

    # Both failed
    raise RuntimeError(
        f"[FY_FLASepformer] Failed to download model.\n"
        f"  Model path: {ONNX_MODEL_PATH}\n"
        f"  Please download manually and place onnx_model.onnx in the directory above.\n"
        f"  ModelScope: https://modelscope.cn/models/iic/speech_flatsepreformer_separation_temporal_8k_base_libri2mix100\n"
        f"  HuggingFace: https://huggingface.co/damo/speech_flatsepreformer_separation_temporal_8k_base_libri2mix100"
    )


# Load ONNX model
_session = None
_provider = None

def get_provider():
    """Get the best available ONNX Runtime provider."""
    available = rt.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return "CUDAExecutionProvider"
    if "ROCMExecutionProvider" in available:
        return "ROCMExecutionProvider"
    return "CPUExecutionProvider"

def get_model_session():
    """Lazy-load the ONNX model session (auto-downloads if missing)."""
    global _session, _provider
    ensure_model_downloaded()
    if _session is None:
        _provider = get_provider()
        print(f"[FY_FLASepformer] Using ONNX Runtime provider: {_provider}")
        _session = rt.InferenceSession(
            ONNX_MODEL_PATH,
            providers=[_provider]
        )
        meta = _session.get_modelmeta()
        print(f"[FY_FLASepformer] Model metadata: {meta.custom_metadata_map}")
    return _session

from .nodes import FY_FLASepformer

NODE_CLASS_MAPPINGS = {
    "FY_FLASepformer": FY_FLASepformer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FY_FLASepformer": "FY FLASepformer",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "ensure_model_downloaded"]
