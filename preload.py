import asyncio

from python.helpers import runtime, whisper, settings
from python.helpers.print_style import PrintStyle
from python.helpers import kokoro_tts
from python.helpers.ram import preload_local_embeddings
import models


async def preload(warm_models: bool | None = None):
    try:
        set = settings.get_default_settings()

        # preload whisper model
        async def preload_whisper():
            try:
                return await whisper.preload(set["stt_model_size"])
            except Exception as e:
                PrintStyle().error(f"Error in preload_whisper: {e}")

        # preload embedding model
        async def preload_embedding():
            if set["embed_model_provider"].lower() == "huggingface":
                try:
                    # Use the new LiteLLM-based model system
                    emb_mod = models.get_embedding_model(
                        "huggingface", set["embed_model_name"]
                    )
                    emb_txt = await emb_mod.aembed_query("test")
                    return emb_txt
                except Exception as e:
                    PrintStyle().error(f"Error in preload_embedding: {e}")

        # preload kokoro tts model if enabled
        async def preload_kokoro():
            if set["tts_kokoro"]:
                try:
                    return await kokoro_tts.preload()
                except Exception as e:
                    PrintStyle().error(f"Error in preload_kokoro: {e}")

        # The running server does not keep the embedding model resident unless
        # A0_PRELOAD_EMBEDDINGS is set. ``python preload.py`` (image build)
        # still passes warm_models=True so the weights land on disk.
        if warm_models is None:
            warm_models = preload_local_embeddings()

        tasks = []
        if warm_models:
            tasks.append(preload_embedding())
            # tasks.append(preload_whisper())
            # tasks.append(preload_kokoro())

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        PrintStyle().print("Preload completed.")
    except Exception as e:
        PrintStyle().error(f"Error in preload: {e}")


# preload transcription model
if __name__ == "__main__":
    PrintStyle().print("Running preload...")
    runtime.initialize()
    # Download weights into the image or local cache, then exit. The process
    # ending is what releases the RAM; the server does not repeat this load.
    asyncio.run(preload(warm_models=True))
