"""
FY_FLASepformer Node - FLASepformer Speech Separation
Separates a mixed audio of two speakers into two independent audio tracks.
"""

import os
import tempfile
import numpy as np
import soundfile as sf
import torch
import torchaudio
import onnxruntime as rt
import folder_paths

# Import model session from __init__ (handled by ComfyUI's module system)
# When loaded by ComfyUI, these are available via the package
try:
    from . import get_model_session, MODEL_DIR, temp_dir
except ImportError:
    # Fallback for direct execution
    import os
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(_current_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    import onnxruntime as rt
    import folder_paths
    MODEL_DIR = os.path.join(
        folder_paths.models_dir,
        "diffusers",
        "speech_flatsepreformer_separation_temporal_8k_base_libri2mix100"
    )
    ONNX_MODEL_PATH = os.path.join(MODEL_DIR, "onnx_model.onnx")

    _session = None
    _provider = None

    def get_provider():
        available = rt.get_available_providers()
        if "CUDAExecutionProvider" in available:
            return "CUDAExecutionProvider"
        if "ROCMExecutionProvider" in available:
            return "ROCMExecutionProvider"
        return "CPUExecutionProvider"

    def get_model_session():
        global _session, _provider
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


class FY_FLASepformer:
    """FLASepformer Speech Separation Node.
    
    Separates a mixed audio of two speakers into two independent audio tracks.
    Input audio is automatically resampled to 8000 Hz mono before inference.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "num_stages": (
                    ["4", "6", "8"],
                    {"default": "4", "tooltip": "Number of separation stages (model default: 4)"}
                ),
                "normalize_output": (
                    ["enabled", "disabled"],
                    {
                        "default": "enabled",
                        "tooltip": "Normalize output audio to prevent clipping (peak scaling to 0.5)"
                    }
                ),
                "use_gpu": (
                    ["auto", "cpu", "gpu"],
                    {
                        "default": "auto",
                        "tooltip": "Use GPU for inference if available (auto: use GPU if ONNX Runtime supports it)"
                    }
                ),
            },
        }

    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("audio_1", "audio_2")
    FUNCTION = "separate"
    CATEGORY = "FY/Audio"
    DESCRIPTION = "FLASepformer: Separate mixed dual-speaker audio into two independent tracks at 8kHz mono."

    def separate(self, audio, num_stages="4", normalize_output="enabled",
                 use_gpu="auto"):
        """Separate mixed audio into two independent speaker tracks."""
        # Get the ONNX model session
        session = get_model_session()

        # Extract audio data
        waveform = audio["waveform"]  # (1, channels, samples)
        sample_rate = audio["sample_rate"]

        # Convert to mono numpy float32
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        # waveform: (channels, samples)

        # Resample to 8000 Hz if needed
        if sample_rate != 8000:
            waveform_8k = torchaudio.functional.resample(
                waveform, sample_rate, 8000
            )
        else:
            waveform_8k = waveform

        # Convert to mono (average channels if stereo)
        if waveform_8k.shape[0] > 1:
            audio_mono = waveform_8k.mean(dim=0, keepdim=True)
        else:
            audio_mono = waveform_8k

        # Ensure float32
        audio_np = audio_mono.squeeze(0).numpy().astype(np.float32)
        input_duration = len(audio_np) / 8000.0
        print(f"[FY_FLASepformer] Input: {len(audio_np)} samples ({input_duration:.2f}s) at 8000 Hz mono")

        # Run inference
        print("[FY_FLASepformer] Running FLASepformer inference...")
        result = session.run(
            None,
            {"mixture": audio_np[None].astype(np.float32)}
        )
        sources = result[0]  # (1, samples, 2)
        print(f"[FY_FLASepformer] Inference complete. Sources shape: {sources.shape}")

        # Process outputs
        outputs = []
        for i in range(2):
            source = sources[0, :, i]  # (samples,)

            # Normalize to prevent clipping
            if normalize_output == "enabled":
                peak = np.max(np.abs(source))
                if peak > 0:
                    source = source * (0.5 / peak)

            # Clip to [-1, 1] range
            source = np.clip(source, -1.0, 1.0)

            # Convert to torch tensor for ComfyUI AUDIO format (batch, channels, time)
            output_tensor = torch.from_numpy(source).unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
            outputs.append({
                "waveform": output_tensor,
                "sample_rate": 8000
            })

        print(f"[FY_FLASepformer] Output: 2 tracks at 8000 Hz mono")
        return (outputs[0], outputs[1])
