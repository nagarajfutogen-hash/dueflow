"""Day 5 WebRTC spike, offline (no browser, no Deepgram, no LLM)."""

import httpx
from pipecat.frames.frames import TranscriptionFrame, TTSSpeakFrame
from pipecat.tests.utils import run_test

from app.models_config import load_models
from app.voice.spike import ECHO_PREFIX, Echo


async def test_echo_speaks_back_final_transcripts() -> None:
    echo = Echo()
    down, _ = await run_test(
        echo,
        frames_to_send=[TranscriptionFrame("hello there", "user", "t0", finalized=True)],
        expected_down_frames=[TTSSpeakFrame],
    )
    spoken = down[0]
    assert isinstance(spoken, TTSSpeakFrame) and spoken.text == f"{ECHO_PREFIX}: hello there"


async def test_echo_does_not_speak_empty_transcripts() -> None:
    # Passed through unchanged; nothing is said.
    await run_test(
        Echo(),
        frames_to_send=[TranscriptionFrame("  ", "user", "t0")],
        expected_down_frames=[TranscriptionFrame],
    )


async def test_offer_needs_the_api_token(api_client: httpx.AsyncClient) -> None:
    body = {"sdp": "v=0", "type": "offer"}
    res = await api_client.post("/voice/spike/offer", json=body, headers={"Authorization": ""})
    assert res.status_code == 401


async def test_offer_rejects_non_offers(api_client: httpx.AsyncClient) -> None:
    res = await api_client.post("/voice/spike/offer", json={"sdp": "v=0", "type": "answer"})
    assert res.status_code == 422


def test_voice_models_come_from_models_yaml() -> None:
    models = load_models()
    assert (models.stt.provider, models.tts.provider) == ("deepgram", "deepgram")
    assert models.stt.model_id and models.tts.model_id
