# Voxtral Test

My investigation of Voxtral Mini-3B's capabilities.

## Installation / Basic operation

On my system `--torch-backend=auto` fails, so I have to manually set it, after querying `nvidia-smi` (see: https://github.com/astral-sh/uv/issues/14647)

```
uv venv --python 3.12 --seed
source .venv/bin/activate

# There is a warning about flashinfer not being available, but I couldn't install it because vllm is on torch 2.7
# uv pip install flashinfer-python -i https://flashinfer.ai/whl/cu126/torch2.6

uv pip install -U "vllm[audio]" --torch-backend=cu126 --extra-index-url https://wheels.vllm.ai/nightly

uv pip install --upgrade mistral_common\[audio\] "numpy<2.3"

# I serve on custom port 8333
vllm serve mistralai/Voxtral-Mini-3B-2507 --port 8333 --tokenizer_mode mistral --config_format mistral --load_format mistral

# Test transcription
uv run transcribe.py

# or 
uv run streaming.py
```

## VAD-based streaming

This script uses `auditok` for Voice Activity Detection (VAD) and streams audio chunks to the Voxtral model for transcription. Takes 8 seconds to transcribe the 180 seconds of Obama Audio on a RTX 3090 or 62 seconds for 31 minutes of the Count of Monte Cristo. Max concurrency is set to 32, but an RTX 3090 is pretty maxed out at ~16 parallel requests.

```
# All the installation before / start vllm / then also:
uv pip install auditok
uv run vad_streaming.py
```

## Timestamps

Voxtral does not supports segment or word level timestamps [their announcement says coming soon](https://mistral.ai/news/voxtral). A poor man's sliding window implementation is provided in `timestamps.py`.

```
uv run timestamps.py
```
