#!/usr/bin/env python
"""
Sliding‑window transcription against a *vLLM OpenAI‑compatible* endpoint **without**
passing any unsupported params.

Changes vs. your original synchronous, non‑overlapping script:

1. **Sliding window:** 1,000 ms window advanced every 100 ms (`STEP_MS`).
2. **Bounded concurrency:** Requests are sent concurrently but *at most* `MAX_CONCURRENCY`
   are in flight to avoid exhausting http connections / file descriptors and to ensure
   the server actually receives traffic. (You saw hangs when launching ~2k simultaneous
   uploads.)
3. **Same request schema** you were already using (via `TranscriptionRequest(...).to_openai()`),
   so we only send the fields your vLLM server expects: `model`, `audio`, `language`, `temperature`.
4. **Word timeline estimation from overlapping chunk transcripts:** Because your endpoint
   does *not* expose per‑word timestamps, we infer when words *enter* and *leave* the rolling
   transcript. We:
      • Tokenize each chunk transcript.
      • Track when each token (case‑insensitive) first appears.
      • Update a token's `last_seen_ms` each time it remains present in later overlapping chunks.
      • Close the token span when it disappears (no longer present in a chunk).
      • Estimated **start_ms** = chunk_start_ms where token first observed.
      • Estimated **end_ms**   = last_chunk_start_ms token was observed + `CHUNK_DURATION_MS`.
        (Because any token present in a chunk could actually be anywhere within that 1s window;
         extending to the end of the window gives a conservative upper bound.)

   This collapses the massive duplication you get from overlapped windows while still allowing
   the *same lexical word* to appear multiple times in the full audio (e.g., "America" early and
   later) because we close the first span before starting a new one.

5. **Ordered output:** We print a table of (start, end, word) in chronological order and a final
   assembled transcript built from the *first spelling we saw* for each span.

You can tune:
   • `MAX_CONCURRENCY` – try 4, 8, 16 depending on server throughput.
   • `TOKEN_PATTERN` – regex for tokenization; default is fairly permissive wordish tokens.
   • `CASE_SENSITIVE` – track separate spans for "Obama" and "obama" (default False → collapsed).

--------------------------------------------------------------------
Quick sanity‑check run (dry):
    uv run time_sliding_async_vllm_safe.py --dry-run
will show you how many chunks will be processed but skip API calls.

Normal run:
    uv run time_sliding_async_vllm_safe.py

--------------------------------------------------------------------
"""

import argparse
import asyncio
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from huggingface_hub import hf_hub_download

from mistral_common.protocol.transcription.request import TranscriptionRequest
from mistral_common.protocol.instruct.messages import RawAudio
from mistral_common.audio import Audio

from openai import AsyncOpenAI

# ---------------- Configuration ----------------
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8333/v1"

CHUNK_DURATION_MS = 1000   # 1s window
STEP_MS = 500             # 0.1s hop
MAX_CONCURRENCY = 64       # <- tune me!
LANGUAGE = "en"
TEMPERATURE = 0.0

# Word tracking config
TOKEN_PATTERN = re.compile(r"[\w']+")  # crude wordish tokens
CASE_SENSITIVE = False                   # lower() tokens for tracking

# ------------------------------------------------


def ms_to_hhmmssmmm(ms: int) -> str:
    s, ms_part = divmod(ms, 1000)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms_part:03d}"


@dataclass
class ChunkSpec:
    idx: int
    start_ms: int
    end_ms: int
    start_sample: int
    end_sample: int


async def transcribe_chunk(
    client: AsyncOpenAI,
    model: str,
    chunk_audio: Audio,
    *,
    language: str,
    temperature: float,
):
    """Send one transcription request; return response object."""
    raw_audio_chunk = RawAudio.from_audio(chunk_audio)
    req = TranscriptionRequest(
        model=model,
        audio=raw_audio_chunk,
        language=language,
        temperature=temperature,
    ).to_openai(exclude=("top_p", "seed"))  # keep the request minimal & compatible

    # NOTE: AsyncOpenAI returns an object; we don't know its exact shape but we expect
    #       `.text` like in your sync client. We'll defensively handle str/dict forms below.
    resp = await client.audio.transcriptions.create(**req)
    return resp


async def worker(
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    model: str,
    audio_data: np.ndarray,
    sample_rate: int,
    audio_format: str,
    spec: ChunkSpec,
    *,
    language: str,
    temperature: float,
):
    """Semaphore‑guarded transcription of one chunk; returns (idx, start_ms, text)."""
    async with sem:
        # Slice without copying if possible (numpy view)
        chunk_data = audio_data[spec.start_sample:spec.end_sample]
        # Construct Audio object
        chunk_audio_obj = Audio(
            audio_array=chunk_data,
            sampling_rate=sample_rate,
            format=audio_format,
        )
        resp = await transcribe_chunk(
            client, model, chunk_audio_obj,
            language=language,
            temperature=temperature,
        )
        # Extract text
        text = getattr(resp, "text", None)
        if text is None:
            # Some implementations put transcript in `resp.data[0].text` or `resp.text` as dict key
            if hasattr(resp, "data") and resp.data:
                maybe = getattr(resp.data[0], "text", None)
                text = maybe if maybe is not None else ""
            elif isinstance(resp, dict):
                text = resp.get("text", "")
            else:
                text = str(resp)
        return spec.idx, spec.start_ms, spec.end_ms, text


def build_chunk_specs(total_samples: int, sample_rate: int) -> List[ChunkSpec]:
    """Prepare the list of chunks for sliding window processing."""
    chunk_size_samples = int(sample_rate * CHUNK_DURATION_MS / 1000)
    step_samples = int(sample_rate * STEP_MS / 1000)

    specs: List[ChunkSpec] = []
    idx = 0
    start_sample = 0
    while start_sample < total_samples:
        end_sample = start_sample + chunk_size_samples
        if end_sample > total_samples:
            end_sample = total_samples
        start_ms = int(start_sample / sample_rate * 1000)
        end_ms = int(end_sample / sample_rate * 1000)
        specs.append(ChunkSpec(idx, start_ms, end_ms, start_sample, end_sample))
        if end_sample >= total_samples:
            break
        start_sample += step_samples
        idx += 1
    return specs


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    toks = TOKEN_PATTERN.findall(text)
    if not CASE_SENSITIVE:
        toks = [t.lower() for t in toks]
    return toks


def build_word_timeline(results: List[Tuple[int, int, int, str]]) -> List[dict]:
    """
    Infer word spans from overlapping chunk transcripts.

    results: list of (idx, start_ms, end_ms, text) sorted by idx.

    We track active tokens in a dict:
        active[token] = {
            'start_ms': first_chunk_start_ms,
            'last_seen_start_ms': most_recent_chunk_start_ms_where_present,
            'display': first_original_form_we_saw
        }

    When token disappears in a chunk, we close it, assigning end_ms = last_seen_start_ms + CHUNK_DURATION_MS.
    Using +CHUNK_DURATION_MS extends span to cover the window in which we last saw the token.
    """
    active: Dict[str, dict] = {}
    closed: List[dict] = []

    for idx, start_ms, end_ms, text in results:
        toks = tokenize(text)
        present = set(toks)
        # Update existing tokens
        for tok in list(active.keys()):
            if tok in present:
                active[tok]['last_seen_start_ms'] = start_ms
            else:
                # close span
                run = active.pop(tok)
                run_end_ms = run['last_seen_start_ms'] + CHUNK_DURATION_MS
                closed.append({
                    'word': run['display'],
                    'start_ms': run['start_ms'],
                    'end_ms': run_end_ms,
                })
        # Add new tokens
        for tok in present:
            if tok not in active:
                # capture original casing from current chunk text: naive search
                # We'll scan text tokens again preserving case
                origs = TOKEN_PATTERN.findall(text)
                if not CASE_SENSITIVE:
                    # pick first match w/ same lowercase
                    orig = next((o for o in origs if o.lower() == tok), tok)
                else:
                    orig = tok
                active[tok] = {
                    'start_ms': start_ms,
                    'last_seen_start_ms': start_ms,
                    'display': orig,
                }

    # close remaining active tokens at end of audio
    if results:
        audio_end_ms = results[-1][2]  # end_ms of last chunk
    else:
        audio_end_ms = 0
    for tok, run in active.items():
        run_end_ms = run['last_seen_start_ms'] + CHUNK_DURATION_MS
        if run_end_ms > audio_end_ms:
            run_end_ms = audio_end_ms
        closed.append({
            'word': run['display'],
            'start_ms': run['start_ms'],
            'end_ms': run_end_ms,
        })

    # sort closed spans by start_ms to ensure chronological order
    closed.sort(key=lambda d: d['start_ms'])
    return closed


def print_timeline(spans: List[dict]):
    print("\n{:<15} {:<15} WORD".format("START", "END"))
    print("-" * 45)
    for s in spans:
        print(f"{ms_to_hhmmssmmm(s['start_ms']):<15} {ms_to_hhmmssmmm(s['end_ms']):<15} {s['word']}")


async def main(dry_run: bool = False):
    # ----- Async client -----
    client = AsyncOpenAI(api_key=openai_api_key, base_url=openai_api_base)
    models = await client.models.list()
    model = models.data[0].id
    print(f"Using model: {model}")

    # ----- Load audio -----
    obama_file = hf_hub_download("patrickvonplaten/audio_samples", "obama.mp3", repo_type="dataset")
    full_audio = Audio.from_file(obama_file, strict=False)
    print(f"Audio loaded: {full_audio}")

    sample_rate = full_audio.sampling_rate
    audio_data = full_audio.audio_array
    audio_format = full_audio.format
    total_samples = len(audio_data)

    specs = build_chunk_specs(total_samples, sample_rate)
    print(f"Prepared {len(specs)} chunks (window={CHUNK_DURATION_MS}ms, step={STEP_MS}ms).")

    if dry_run:
        print("Dry run requested; no API calls made.")
        return

    # ----- Fire off bounded concurrent requests -----
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks = [
        asyncio.create_task(
            worker(
                sem,
                client,
                model,
                audio_data,
                sample_rate,
                audio_format,
                spec,
                language=LANGUAGE,
                temperature=TEMPERATURE,
            )
        )
        for spec in specs
    ]

    print(f"Submitting transcription tasks (max {MAX_CONCURRENCY} in flight)…")
    results = await asyncio.gather(*tasks)

    # sort by chunk idx just to be safe (gather preserves order of task creation, but defensive)
    results = sorted(results, key=lambda x: x[0])

    # ----- Build word timeline -----
    spans = build_word_timeline(results)

    # ----- Print results -----
    print_timeline(spans)

    # joined text
    final_text = " ".join(s['word'] for s in spans)
    print("\nFinal joined transcription:")
    print(final_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sliding window transcription (vLLM compatible)")
    parser.add_argument("--dry-run", action="store_true", help="Do not call the API; just show chunk stats")
    args = parser.parse_args()

    asyncio.run(main(dry_run=args.dry_run))
