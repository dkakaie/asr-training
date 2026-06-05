import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import librosa
import nemo.collections.asr as nemo_asr
import soundfile

TARGET_LANG = "he-IL"
SAMPLE_RATE = 16_000
# Right context 13 → 1.12s chunk size, highest-accuracy setting per the model card.
# Other supported values: 0 (80ms), 1 (160ms), 3 (320ms), 6 (560ms), 13 (1.12s).
ATT_CONTEXT_SIZE = [56, 13]
LANG_TAG_RE = re.compile(r"\s*<[a-z]{2}-[A-Z]{2}>\s*")


def _extract_text(result) -> str:
    item = result[0] if isinstance(result, list) else result
    if isinstance(item, list):
        item = item[0]
    text = item.text if hasattr(item, "text") else str(item)
    return LANG_TAG_RE.sub(" ", text).strip()


def _entry_to_wav_16k_mono(entry: Dict[str, Any]) -> Tuple[Path, float]:
    audio = entry["audio"]["array"]
    sr = entry["audio"]["sampling_rate"]
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    fd, tmp_path = tempfile.mkstemp(prefix="nemo_asr_", suffix=".wav")
    os.close(fd)
    soundfile.write(tmp_path, audio, SAMPLE_RATE, subtype="PCM_16")
    return Path(tmp_path), len(audio) / SAMPLE_RATE


def transcribe(model, entry: Dict[str, Any]) -> Tuple[str, float]:
    wav_path, duration = _entry_to_wav_16k_mono(entry)
    fd, manifest_path = tempfile.mkstemp(prefix="nemo_asr_", suffix=".json")
    os.close(fd)
    try:
        manifest_entry = {
            "audio_filepath": str(wav_path),
            "duration": duration,
            "text": "",
            "lang": TARGET_LANG,
        }
        with open(manifest_path, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(manifest_entry) + "\n")

        start_time = time.time()
        result = model.transcribe([manifest_path], batch_size=1, target_lang=TARGET_LANG)
        transcription_time = time.time() - start_time
    finally:
        wav_path.unlink(missing_ok=True)
        Path(manifest_path).unlink(missing_ok=True)

    return _extract_text(result), transcription_time


def create_app(**kwargs) -> Callable:
    model_path = kwargs.get("model_path")
    device: str = kwargs.get("device", "auto")

    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading nemotron-asr model: {model_path} on {device}")
    if model_path and os.path.isfile(model_path):
        model = nemo_asr.models.ASRModel.restore_from(str(model_path), map_location=device)
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=model_path or "nvidia/nemotron-3.5-asr-streaming-0.6b",
            map_location=device,
        )
    model.eval()

    if hasattr(model.encoder, "set_default_att_context_size"):
        model.encoder.set_default_att_context_size(att_context_size=ATT_CONTEXT_SIZE)
    if hasattr(model, "decoding") and hasattr(model.decoding, "set_strip_lang_tags"):
        model.decoding.set_strip_lang_tags(True)

    def transcribe_fn(entries):
        if not isinstance(entries, list):
            entries = [entries]
        return [transcribe(model, entry) for entry in entries]

    return transcribe_fn
