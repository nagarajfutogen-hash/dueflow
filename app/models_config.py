"""Loads model IDs from config/models.yaml (committed). Secrets come from .env, never from here."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_MODELS_PATH = Path(__file__).resolve().parent.parent / "config" / "models.yaml"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ModelRef(_Frozen):
    provider: Literal["groq", "openai", "deepgram"]
    model_id: str = Field(min_length=1)


class ModelsConfig(_Frozen):
    agent: ModelRef
    judge: ModelRef
    simulator: ModelRef
    stt: ModelRef  # voice: Deepgram speech-to-text
    tts: ModelRef  # voice: Deepgram text-to-speech (model_id = the voice)


def load_models(path: Path = DEFAULT_MODELS_PATH) -> ModelsConfig:
    return ModelsConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
