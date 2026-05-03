import io
import os
import time
from typing import Callable

import soundfile
from dotenv import load_dotenv
from retry import retry
from soniox.client import SonioxClient


def create_app(**kwargs) -> Callable:
    load_dotenv()

    model_path = kwargs.get("model_path", "")

    print("Initializing Soniox client")

    api_key = os.environ["SONIOX_API_KEY"]
    client = SonioxClient(api_key=api_key)

    @retry(tries=3, delay=1, backoff=2)
    def transcribe_one(entry):
        wav_buffer = io.BytesIO()
        soundfile.write(
            wav_buffer,
            entry["audio"]["array"],
            entry["audio"]["sampling_rate"],
            format="WAV",
        )
        wav_buffer.seek(0)

        try:
            start_time = time.time()
            result = client.stt.transcribe_and_wait_with_tokens(
                file=wav_buffer,
                delete_after=True,
            )
            transcription_time = time.time() - start_time

            text = "".join(token.text for token in result.tokens)
            return text, transcription_time
        except Exception as e:
            print(f"Exception calling Soniox API: {e}")
            raise e

    def transcribe(entries):
        if not isinstance(entries, list):
            entries = [entries]
        return [transcribe_one(entry) for entry in entries]

    return transcribe
