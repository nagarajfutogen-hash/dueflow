"""Spike self-test, server side only (dev): a WebRTC client inside the container speaks a
Deepgram-synthesized test sentence and records the echo, then Deepgram transcribes the echo.

Proves WebRTC (aiortc) + Deepgram STT -> echo -> TTS in this container. It does NOT prove the
browser -> Docker Desktop network path: that is the manual browser test.

  docker compose exec api python -m app.voice.spike_selftest
"""

import asyncio
import io
import os
import tempfile
import wave
from pathlib import Path

import httpx
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer, MediaRecorder

from app.config import load_settings
from app.models_config import load_models

SENTENCE = "Hello, this is a WebRTC test."  # synthetic test text, not debtor data
SAMPLE_RATE = 48_000
TAIL_SILENCE_S = 3.0  # lets the server-side STT finalize the utterance
LISTEN_S = 15.0
HTTP_TIMEOUT = httpx.Timeout(30.0)


def _with_silence(wav_bytes: bytes) -> bytes:
    with wave.open(io.BytesIO(wav_bytes)) as src:
        params, frames = src.getparams(), src.readframes(src.getnframes())
    pad = b"\0" * int(TAIL_SILENCE_S * params.framerate) * params.sampwidth * params.nchannels
    out = io.BytesIO()
    with wave.open(out, "wb") as dst:
        dst.setparams(params)
        dst.writeframes(frames + pad)
    return out.getvalue()


async def _speak(client: httpx.AsyncClient, key: str, voice: str) -> bytes:
    res = await client.post(
        "https://api.deepgram.com/v1/speak",
        params={
            "model": voice,
            "encoding": "linear16",
            "sample_rate": SAMPLE_RATE,
            "container": "wav",
        },  # fmt: skip
        headers={"Authorization": f"Token {key}"},
        json={"text": SENTENCE},
    )
    res.raise_for_status()
    return res.content


async def _transcribe(client: httpx.AsyncClient, key: str, model: str, wav: bytes) -> str:
    res = await client.post(
        "https://api.deepgram.com/v1/listen",
        params={"model": model, "smart_format": "true"},
        headers={"Authorization": f"Token {key}", "Content-Type": "audio/wav"},
        content=wav,
    )
    res.raise_for_status()
    alt = res.json()["results"]["channels"][0]["alternatives"][0]
    return str(alt.get("transcript", ""))


async def main() -> int:
    settings, models = load_settings(), load_models()
    key = settings.deepgram_api_key
    if not key or not settings.api_token:
        print("DEEPGRAM_API_KEY and API_TOKEN must be set")
        return 2
    work = Path(tempfile.mkdtemp(prefix="spike-"))
    said, heard = work / "said.wav", work / "heard.wav"
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        audio = _with_silence(await _speak(client, key, models.tts.model_id))
        await asyncio.to_thread(said.write_bytes, audio)

        pc = RTCPeerConnection()
        player = MediaPlayer(str(said))
        recorder = MediaRecorder(str(heard))
        assert player.audio is not None, "synthesized file has no audio track"
        pc.addTrack(player.audio)
        pc.addTransceiver("video", direction="recvonly")
        state: dict[str, str] = {}

        @pc.on("track")
        def _on_track(track: MediaStreamTrack) -> None:
            if track.kind == "audio":
                recorder.addTrack(track)

        @pc.on("iceconnectionstatechange")
        def _on_ice() -> None:
            state["ice"] = pc.iceConnectionState

        await pc.setLocalDescription(await pc.createOffer())  # aiortc gathers fully
        res = await client.post(
            f"http://localhost:{os.environ.get('PORT', '8000')}/voice/spike/offer",
            headers={"Authorization": f"Bearer {settings.api_token}"},
            json={"sdp": pc.localDescription.sdp, "type": "offer"},
        )
        res.raise_for_status()
        await pc.setRemoteDescription(RTCSessionDescription(**res.json()))
        await recorder.start()
        await asyncio.sleep(LISTEN_S)
        await recorder.stop()
        await pc.close()

        echo = await asyncio.to_thread(heard.read_bytes)
        transcript = await _transcribe(client, key, models.stt.model_id, echo)
    ok = "heard" in transcript.lower() and "test" in transcript.lower()
    print(f"ice: {state.get('ice', 'unknown')}")
    print(f"echo transcript: {transcript!r}")
    print("SPIKE SERVER SIDE: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
