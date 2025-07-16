from mistral_common.protocol.transcription.request import TranscriptionRequest
from mistral_common.protocol.instruct.messages import RawAudio
from mistral_common.audio import Audio
from huggingface_hub import hf_hub_download

from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8333/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

obama_file = hf_hub_download("patrickvonplaten/audio_samples", "obama.mp3", repo_type="dataset")
audio = Audio.from_file(obama_file, strict=False)

audio = RawAudio.from_audio(audio)
req = TranscriptionRequest(model=model, audio=audio, language="en", temperature=0.0).to_openai(exclude=("top_p", "seed"), response_format="json", stream=True)

print(req)
# Send the transcription request to the vLLM server.

response = client.audio.transcriptions.create(**req)

for chunk in response:
    if chunk.choices:
        content = chunk.choices[0].get("delta", {}).get("content")
        print(content, end="", flush=True)
