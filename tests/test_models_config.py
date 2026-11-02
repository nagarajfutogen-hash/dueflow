from pathlib import Path

import pytest
from pydantic import ValidationError

from app.models_config import load_models

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_models_yaml_loads() -> None:
    models = load_models()
    assert models.agent.model_id
    assert models.judge.model_id


def test_judge_is_not_openai_family() -> None:
    # PLANNING §2: the judge must be a different family from the (final) OpenAI agent.
    models = load_models()
    assert not models.judge.model_id.startswith("openai/")
    assert models.judge.provider != "openai"


def test_unknown_provider_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "models.yaml"
    bad.write_text(
        "agent: {provider: acme, model_id: x}\njudge: {provider: groq, model_id: y}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_models(bad)


def test_model_ids_not_in_env_files() -> None:
    # .env holds secrets only; model IDs live in config/models.yaml.
    example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "MODEL_ID" not in example


def test_judge_never_grades_its_own_model() -> None:
    models = load_models()
    assert models.judge.model_id not in {models.agent.model_id, models.simulator.model_id}


def test_agent_key_follows_the_agent_provider() -> None:
    from app.config import Settings

    s = Settings("u", "p", "db", "h", 5432, "a", "ap", groq_api_key="G", openai_api_key="O")
    assert s.agent_api_key("openai") == "O"
    assert s.agent_api_key("groq") == "G"  # never the OpenAI key for a Groq model
