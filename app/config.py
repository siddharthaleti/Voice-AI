"""
Application configuration for Vaani Duplex using pydantic-settings.
"""

from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized configuration settings loaded from environment or .env file.
    """
    # API Keys
    groq_api_key: str = "gsk_paste_your_free_groq_key_here"
    sarvam_api_key: str = ""

    # Pipeline selection: 'free' or 'sarvam'
    speech_provider: Literal["free", "sarvam"] = "free"

    # LLM Models
    groq_primary_model: str = "llama-3.3-70b-versatile"
    groq_fallback_model: str = "llama-3.1-8b-instant"

    # Silero VAD parameters
    vad_threshold: float = 0.65
    barge_in_min_frames: int = 6       # 192ms (6 * 32ms) continuous speech for interruption
    silence_end_frames: int = 18       # 576ms (18 * 32ms) trailing silence for turn completion
    speech_start_min_frames: int = 3   # 96ms (3 * 32ms) continuous speech to trigger speech start
    pre_roll_frames: int = 5           # 160ms (5 * 32ms) audio pre-roll buffer

    # Audio specifications
    sample_rate: int = 16000           # 16 kHz mono
    frame_samples: int = 512           # 512 samples per frame (32ms at 16kHz)
    bytes_per_sample: int = 2          # 16-bit Int16 PCM (2 bytes)

    @property
    def frame_bytes(self) -> int:
        """Expected byte length per raw PCM frame."""
        return self.frame_samples * self.bytes_per_sample  # 1024 bytes

    @property
    def is_groq_configured(self) -> bool:
        """Check if Groq API key is validly provided."""
        return bool(
            self.groq_api_key
            and self.groq_api_key.strip() != "gsk_paste_your_free_groq_key_here"
            and not self.groq_api_key.startswith("gsk_paste")
        )

    @property
    def is_sarvam_configured(self) -> bool:
        """Check if Sarvam API key is provided."""
        return bool(self.sarvam_api_key and len(self.sarvam_api_key.strip()) > 5)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    """Return cached instance of application settings."""
    return Settings()
