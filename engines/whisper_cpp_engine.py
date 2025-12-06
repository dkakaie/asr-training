import io
import tempfile
import time
from typing import Any, Callable, Dict, Tuple

import ivrit
import pydub
import soundfile


def transcribe(model, entry: Dict[str, Any]) -> Tuple[str, float]:
    """Transcribe audio using Ivrit whisper-cpp engine."""
    # Convert audio to MP3 format
    wav_buffer = io.BytesIO()
    soundfile.write(wav_buffer, entry["audio"]["array"], entry["audio"]["sampling_rate"], format="WAV")
    wav_buffer.seek(0)

    # Convert WAV to MP3
    audio = pydub.AudioSegment.from_file(wav_buffer, format="wav")
    
    # Create a temporary MP3 file (will be cleaned up on exit)
    with tempfile.NamedTemporaryFile(suffix=".mp3") as temp_mp3:
        temp_path = temp_mp3.name
        audio.export(temp_path, format="mp3")
        
        start_time = time.time()
        
        # Use transcribe() and iterate over segments
        res = model.transcribe(path=temp_path, language="he")
                
        transcription_time = time.time() - start_time
        
        return res['text'], transcription_time


def get_device_and_index(device: str) -> tuple[str, int | None]:
    """Parse device string to extract device type and index."""
    if len(device.split(":")) == 2:
        device, device_index = device.split(":")
        device_index = int(device_index)
        return device, device_index

    return device, None


def create_app(**kwargs) -> Callable:
    """Create the Ivrit whisper-cpp transcription application."""
    model_path = kwargs.get("model_path")
    device: str = kwargs.get("device", "auto")
    device_index = None

    if not model_path:
        raise ValueError("model_path is required for whisper-cpp engine")

    # Handle multiple devices (similar to faster_whisper_engine)
    if len(device.split(",")) > 1:
        device_indexes = []
        base_device = None
        for device_instance in device.split(","):
            device, device_index = get_device_and_index(device_instance)
            base_device = base_device or device
            if base_device != device:
                raise ValueError("Multiple devices must be instances of the same base device (e.g cuda:0, cuda:1 etc.)")
            device_indexes.append(device_index)
        device = base_device
        device_index = device_indexes
    else:
        device, device_index = get_device_and_index(device)

    # Prepare arguments for ivrit.load_model
    load_args = {'engine': 'whisper-cpp', 'model': model_path}
    
    # Add device configuration if specified
    if device != "auto":
        load_args['device'] = device
        if device_index is not None:
            load_args['device_index'] = device_index

    print(f'Loading ivrit whisper-cpp model: {model_path} on {device} with index: {device_index or 0}')
    
    # Load the model using ivrit
    model = ivrit.load_model(**load_args)

    def transcribe_fn(entry):
        return transcribe(model, entry)

    return transcribe_fn

