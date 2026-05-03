import io
import os
import time
from typing import Callable

import soundfile
from dotenv import load_dotenv
from deepgram import DeepgramClient


def create_app(**kwargs) -> Callable:
    load_dotenv()

    model_path = kwargs.get("model_path", "nova-3")

    print(f"Initializing Deepgram client, model: {model_path}")

    api_key = os.environ["DEEPGRAM_API_KEY"]
    client = DeepgramClient(api_key=api_key)

    def transcribe_one(entry):
        wav_buffer = io.BytesIO()
        soundfile.write(
            wav_buffer,
            entry["audio"]["array"],
            entry["audio"]["sampling_rate"],
            format="WAV",
        )
        audio_bytes = wav_buffer.getvalue()

        try:
            start_time = time.time()
            response = client.listen.v1.media.transcribe_file(
                request=audio_bytes,
                model=model_path,
                language="he",
            )
            transcription_time = time.time() - start_time

            text = response.results.channels[0].alternatives[0].transcript or ""
            return text, transcription_time
        except Exception as e:
            print(f"Exception calling Deepgram API: {e}")
            raise e

    def transcribe(entries):
        if not isinstance(entries, list):
            entries = [entries]
        return [transcribe_one(entry) for entry in entries]

    return transcribe
