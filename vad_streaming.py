#!/usr/bin/env python
"""
VAD-based, concurrent, streaming transcription against a vLLM OpenAI-compatible endpoint.

This script operates in two distinct stages for maximum reliability:
1.  **Process and Save:** It uses Voice Activity Detection (VAD) to find speech,
    processes each chunk (downmix, resample), and saves it to disk as a WAV file
    in the `processed_chunks/` directory.
2.  **Load and Transcribe:** It then loads each saved WAV file from disk and
    submits it for transcription, streaming the results to the console.

This file-based approach eliminates in-memory data corruption issues and allows for
easy inspection of the exact audio data being sent to the API.
"""
import asyncio
from pathlib import Path
import tempfile
from typing import Tuple, AsyncGenerator
import os

from huggingface_hub import hf_hub_download
import auditok

from mistral_common.protocol.transcription.request import TranscriptionRequest
from mistral_common.protocol.instruct.messages import RawAudio
from mistral_common.audio import Audio

from openai import AsyncOpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8333/v1"

MAX_CONCURRENCY = 32

# VAD (Voice Activity Detection) parameters
VAD_ENERGY_THRESHOLD = 55
VAD_MIN_DUR_S = 0.2
VAD_MAX_DUR_S = 30.0
VAD_SILENCE_S = 0.5

LANGUAGE = "en"

def ms_to_hhmmssmmm(ms: int) -> str:
    s, ms_part = divmod(ms, 1000)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms_part:03d}"

async def process_and_transcribe_chunk(
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    model: str,
    region: auditok.AudioRegion
) -> Tuple[int, int, AsyncGenerator[str, None]]:
    """Saves an audio region and returns streaming generator for transcription."""
    
    await sem.acquire()

    # Ugly to do this round-trip, but otherwise we lose all the metadata
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_wav = Path(tmpdir) / 'chunk.wav'
        region.save(tmp_wav)
        raw_audio_chunk = RawAudio.from_audio(Audio.from_file(tmp_wav, strict=False))

    req = TranscriptionRequest(
        model=model,
        audio=raw_audio_chunk,
        language=LANGUAGE,
        temperature=0.0,
    ).to_openai(
        exclude=("top_p", "seed"),
        response_format="json",
        stream=True,
    )

    start_ms = int(region.start * 1000)
    end_ms = int(region.end * 1000)
    
    response = await client.audio.transcriptions.create(**req)
    
    async def content_generator():
        async for chunk in response:
            if chunk.choices:
                yield chunk.choices[0].get("delta", {}).get("content")
        
        sem.release()
    
    return (start_ms, end_ms, content_generator())


async def process_and_transcribe_streaming(audio_path: str, client: AsyncOpenAI, model: str):
    """
    Splits audio with VAD and processes chunks in parallel, with ordered output.
    """
    print("Processing audio with VAD and streaming transcription...")

    audio_regions = auditok.split(
        audio_path,
        min_dur=VAD_MIN_DUR_S, 
        max_dur=VAD_MAX_DUR_S,
        max_silence=VAD_SILENCE_S, 
        energy_threshold=VAD_ENERGY_THRESHOLD,
    )

    # Process and transcribe in parallel
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    
    # Create tasks for parallel processing (start all immediately)
    tasks = [asyncio.create_task(process_and_transcribe_chunk(sem, client, model, region)) for region in audio_regions]
    
    print(f"Dispatching {len(tasks)} chunks for parallel processing (max {MAX_CONCURRENCY} concurrent)...")
    
    # Await results in order and stream content as each becomes available
    final_texts = []
    for task in tasks:
        start_ms, end_ms, content_gen = await task
        
        print(f"\n[{ms_to_hhmmssmmm(start_ms)} -> {ms_to_hhmmssmmm(end_ms)}] ", end="", flush=True)        
        async for content in content_gen:
            final_texts.append(content)
            print(content, end="", flush=True)
        final_texts.append(" ")
    
    print("\n\nFinal Assembled Transcription:")
    print("".join(final_texts))

async def main():
    client = AsyncOpenAI(api_key=openai_api_key, base_url=openai_api_base)
    models = await client.models.list()
    model = models.data[0].id
    print(f"Using model: {model}")

    # obama_file_path = hf_hub_download("patrickvonplaten/audio_samples", "obama.mp3", repo_type="dataset")
    # children_story = hf_hub_download("ajibawa-2023/Audio-Children-Stories-Collection", "story_318.mp3", repo_type="dataset")
    await process_and_transcribe_streaming("count_of_monte_cristo_003_dumas.mp3", client, model)

if __name__ == "__main__":
    asyncio.run(main())