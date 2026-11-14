import base64
import os
import warnings
import tempfile
import asyncio
from python.helpers import runtime, rfc, settings, files
from python.helpers.print_style import PrintStyle
from python.helpers.notification import NotificationManager, NotificationType, NotificationPriority
from python.helpers.ram import IdleGuard, configure_torch_threads, local_model_idle_seconds

# Suppress FutureWarning from torch.load
warnings.filterwarnings("ignore", category=FutureWarning)

# openai-whisper imports torch. Load the package on first transcription so an
# idle UI process does not pay for it.

_model = None
_model_name = ""
is_updating_model = False  # Tracks whether the model is currently updating


def _whisper_lib():
    import whisper

    return whisper


def _drop_model() -> None:
    global _model, _model_name
    if _model is None:
        return
    _model = None
    _model_name = ""
    PrintStyle.standard("Released Whisper model after idle.")


_residency = IdleGuard(local_model_idle_seconds(), _drop_model)

async def preload(model_name:str):
    try:
        # return await runtime.call_development_function(_preload, model_name)
        return await _preload(model_name)
    except Exception as e:
        # if not runtime.is_development():
        raise e
        
async def _preload(model_name:str):
    global _model, _model_name, is_updating_model

    _residency.acquire()
    try:
        while is_updating_model:
            await asyncio.sleep(0.1)

        try:
            is_updating_model = True
            if not _model or _model_name != model_name:
                NotificationManager.send_notification(
                    NotificationType.INFO,
                    NotificationPriority.NORMAL,
                    "Loading Whisper model...",
                    display_time=99,
                    group="whisper-preload")
                PrintStyle.standard(f"Loading Whisper model: {model_name}")
                configure_torch_threads()
                _model = _whisper_lib().load_model(name=model_name, download_root=files.get_abs_path("/tmp/models/whisper")) # type: ignore
                _model_name = model_name
                NotificationManager.send_notification(
                    NotificationType.INFO,
                    NotificationPriority.NORMAL,
                    "Whisper model loaded.",
                    display_time=2,
                    group="whisper-preload")
        finally:
            is_updating_model = False
    finally:
        _residency.release()

async def is_downloading():
    # return await runtime.call_development_function(_is_downloading)
    return _is_downloading()

def _is_downloading():
    return is_updating_model

async def is_downloaded():
    try:
        # return await runtime.call_development_function(_is_downloaded)
        return _is_downloaded()
    except Exception as e:
        # if not runtime.is_development():
        raise e
        # Fallback to direct execution if RFC fails in development
        # return _is_downloaded()

def _is_downloaded():
    return _model is not None

async def transcribe(model_name:str, audio_bytes_b64: str):
    # return await runtime.call_development_function(_transcribe, model_name, audio_bytes_b64)
    return await _transcribe(model_name, audio_bytes_b64)


async def _transcribe(model_name:str, audio_bytes_b64: str):
    # Hold the model across preload and transcribe so the idle timer cannot
    # drop it between the two steps.
    _residency.acquire()
    try:
        await _preload(model_name)
        model = _model
        if model is None:
            raise RuntimeError("Whisper model is not loaded")

        # Decode audio bytes if encoded as a base64 string
        audio_bytes = base64.b64decode(audio_bytes_b64)

        # Create temp audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
            audio_file.write(audio_bytes)
            temp_path = audio_file.name
        try:
            # Transcribe the audio file. Keep the local reference so an idle
            # drop of the global cannot free the weights mid-call.
            result = model.transcribe(temp_path, fp16=False) # type: ignore
            return result
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass # ignore errors during cleanup
    finally:
        _residency.release()
