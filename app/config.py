import os
from dataclasses import dataclass

from sqlalchemy import URL


@dataclass(frozen=True)
class Settings:
    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_host: str
    postgres_port: int
    app_db_user: str
    app_db_password: str
    groq_api_key: str = ""
    openai_api_key: str = ""
    deepgram_api_key: str = ""
    api_token: str = ""
    admin_password: str = ""

    def _url(self, username: str, password: str) -> URL:
        return URL.create(
            "postgresql+asyncpg",
            username=username,
            password=password,
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        )

    def agent_api_key(self, provider: str) -> str:
        """Key for the agent LLM's provider (config/models.yaml agent.provider)."""
        return self.openai_api_key if provider == "openai" else self.groq_api_key

    @property
    def owner_url(self) -> URL:
        """Schema owner: migrations only."""
        return self._url(self.postgres_user, self.postgres_password)

    @property
    def app_url(self) -> URL:
        """Restricted role the API runs as (no UPDATE/DELETE on events)."""
        return self._url(self.app_db_user, self.app_db_password)


def load_settings() -> Settings:
    return Settings(
        postgres_user=os.environ["POSTGRES_USER"],
        postgres_password=os.environ["POSTGRES_PASSWORD"],
        postgres_db=os.environ["POSTGRES_DB"],
        postgres_host=os.environ.get("POSTGRES_HOST", "postgres"),
        postgres_port=int(os.environ.get("POSTGRES_PORT", "5432")),
        app_db_user=os.environ["APP_DB_USER"],
        app_db_password=os.environ["APP_DB_PASSWORD"],
        groq_api_key=os.environ.get("GROQ_API_KEY", ""),
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        deepgram_api_key=os.environ.get("DEEPGRAM_API_KEY", ""),
        api_token=os.environ.get("API_TOKEN", ""),
        admin_password=os.environ.get("ADMIN_PASSWORD", ""),
    )
