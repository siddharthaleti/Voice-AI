"""
Speech Pipeline Providers for Vaani Duplex.
Supports:
1. Free Mode (₹0): Groq Whisper-large-v3-turbo (ASR) + Microsoft Edge-TTS (TTS)
2. Sarvam AI Mode: Saaras:v4 (ASR) + Bulbul:v3 (TTS)
Includes dynamic fallback from Sarvam to Free Mode on errors.
"""

import base64
import io
import logging
import re
import wave
from typing import Optional, Tuple

import edge_tts
import httpx
from groq import AsyncGroq

from app.config import get_settings

logger = logging.getLogger("vaani.providers")

# Indic Unicode regex blocks
TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

# Common Romanized Hinglish vocabulary tokens
HINGLISH_TOKENS = {
    "kya", "karo", "karenge", "hoga", "nahi", "chahiye", "aap", "hai",
    "mera", "meri", "mere", "theek", "kaise", "namaste", "shukriya",
    "kripya", "bolo", "batao", "samajh", "aaya", "ticket", "order",
    "meeting", "rakho", "dikhao", "chalega", "badhiya", "dhanyavaad"
}

# Common Romanized Telugish vocabulary tokens
TELUGISH_TOKENS = {
    "emi", "ela", "undi", "chesi", "cheyandi", "avunu", "kadu", "meeru",
    "naa", "bagundi", "dhanyavadalu", "cheppandi", "eppudu", "ekkada",
    "kavalenu", "chesara", "ayindi", "vaddu", "gurinchi", "choodandi"
}

# Edge-TTS voice mappings
EDGE_TTS_VOICES = {
    "hi-IN": "hi-IN-SwaraNeural",
    "te-IN": "te-IN-ShrutiNeural",
    "en-IN": "en-IN-NeerjaNeural",
}

# Sarvam Bulbul:v3 speaker mappings
SARVAM_SPEAKERS = {
    "hi-IN": "priya",
    "te-IN": "kavya",
    "en-IN": "ritu",
}


def pcm16_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
    """Wraps raw 16-bit PCM mono bytes into a valid WAV container."""
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return wav_io.getvalue()


def detect_indic_language(text: str, hint: str = "en-IN") -> str:
    """
    Detects whether the text is Hindi, Telugu, or Indian English based on
    Unicode script ranges and romanized lexical heuristics.
    """
    if not text:
        return hint

    # 1. Native script detection
    telugu_count = len(TELUGU_RE.findall(text))
    devanagari_count = len(DEVANAGARI_RE.findall(text))

    if telugu_count > 0 and telugu_count >= devanagari_count:
        return "te-IN"
    if devanagari_count > 0:
        return "hi-IN"

    # 2. Romanized code-switching detection (Hinglish / Telugish in Latin script)
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if words:
        word_set = set(words)
        hinglish_hits = len(word_set.intersection(HINGLISH_TOKENS))
        telugish_hits = len(word_set.intersection(TELUGISH_TOKENS))

        if telugish_hits > 0 and telugish_hits >= hinglish_hits:
            return "te-IN"
        if hinglish_hits > 0:
            return "hi-IN"

    # 3. Fallback to passed hint or default Indian English
    if hint in ("hi-IN", "te-IN", "en-IN"):
        return hint
    return "en-IN"


def clean_text_for_speech(text: str) -> str:
    """Strips markdown asterisks, hashes, backticks, and emojis before synthesis."""
    # Remove markdown formatting
    cleaned = re.sub(r"[\*\_#`~]", "", text)
    # Remove URLs if any
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    # Remove emoji characters
    cleaned = re.sub(r"[\U00010000-\U0010ffff]", "", cleaned)
    # Collapse extra whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


class SpeechPipelineProviders:
    """
    Handles speech-to-text and text-to-speech across Groq (free) and Sarvam AI.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    async def transcribe(
        self,
        pcm_bytes: bytes,
        provider: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Transcribes PCM audio. Returns (transcript_text, detected_language_code).
        Falls back to Groq Whisper if Sarvam AI is unavailable or encounters an error.
        """
        active_provider = provider or self.settings.speech_provider
        wav_bytes = pcm16_to_wav_bytes(pcm_bytes, self.settings.sample_rate)

        # 1. Try Sarvam AI if requested and key is configured
        if active_provider == "sarvam" and self.settings.is_sarvam_configured:
            try:
                transcript, lang = await self._transcribe_sarvam(wav_bytes)
                if transcript:
                    return transcript, lang
            except Exception as e:
                logger.warning(
                    "Sarvam STT failed (%s). Falling back seamlessly to Groq Whisper...", e
                )

        # 2. Free Mode / Fallback: Groq Whisper-large-v3-turbo
        return await self._transcribe_groq(wav_bytes)

    async def _transcribe_sarvam(self, wav_bytes: bytes) -> Tuple[str, str]:
        """Transcribe using Sarvam Saaras:v4 API."""
        url = "https://api.sarvam.ai/speech-to-text"
        headers = {"api-subscription-key": self.settings.sarvam_api_key.strip()}

        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {
            "model": "saaras:v4",
            "mode": "codemix",
            "language_code": "unknown",
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, headers=headers, files=files, data=data)
            response.raise_for_status()
            result = response.json()

        transcript = result.get("transcript", "").strip()
        raw_lang = result.get("language_code", "")

        # Standardize language code
        if "hi" in raw_lang:
            lang_code = "hi-IN"
        elif "te" in raw_lang:
            lang_code = "te-IN"
        elif "en" in raw_lang:
            lang_code = "en-IN"
        else:
            lang_code = detect_indic_language(transcript)

        return transcript, lang_code

    async def _transcribe_groq(self, wav_bytes: bytes) -> Tuple[str, str]:
        """Transcribe using Groq Whisper-large-v3-turbo (100% Free)."""
        groq_key = self.settings.groq_api_key.strip()
        client = AsyncGroq(api_key=groq_key)

        prompt = (
            "Hinglish, Telugish, Hindi, Telugu, Indian English conversation. "
            "Preserve code-switching accurately."
        )

        response = await client.audio.transcriptions.create(
            file=("audio.wav", wav_bytes, "audio/wav"),
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
            prompt=prompt,
            temperature=0.0,
        )

        transcript = getattr(response, "text", "").strip()
        detected_lang = getattr(response, "language", "english").lower()

        # Map Whisper language output
        if detected_lang in ("hi", "hindi"):
            mapped_lang = "hi-IN"
        elif detected_lang in ("te", "telugu"):
            mapped_lang = "te-IN"
        else:
            mapped_lang = "en-IN"

        # Refine with Indic lexical heuristics
        final_lang = detect_indic_language(transcript, hint=mapped_lang)
        return transcript, final_lang

    async def synthesize_clause(
        self,
        text: str,
        lang_code: str,
        provider: Optional[str] = None
    ) -> Tuple[bytes, str]:
        """
        Synthesizes a short clause of text to audio bytes.
        Returns (audio_bytes, content_type).
        Falls back to Edge-TTS if Sarvam AI is unavailable or fails.
        """
        cleaned_text = clean_text_for_speech(text)
        if not cleaned_text:
            return b"", "audio/mpeg"

        active_provider = provider or self.settings.speech_provider

        # 1. Try Sarvam Bulbul:v3 if requested and key is configured
        if active_provider == "sarvam" and self.settings.is_sarvam_configured:
            try:
                audio_bytes, mime = await self._synthesize_sarvam(cleaned_text, lang_code)
                if audio_bytes:
                    return audio_bytes, mime
            except Exception as e:
                logger.warning(
                    "Sarvam TTS failed (%s). Falling back seamlessly to Edge-TTS...", e
                )

        # 2. Free Mode / Fallback: Microsoft Edge-TTS
        return await self._synthesize_edge_tts(cleaned_text, lang_code)

    async def _synthesize_sarvam(
        self, text: str, lang_code: str
    ) -> Tuple[bytes, str]:
        """Synthesize using Sarvam Bulbul:v3 API."""
        url = "https://api.sarvam.ai/text-to-speech"
        headers = {
            "api-subscription-key": self.settings.sarvam_api_key.strip(),
            "Content-Type": "application/json",
        }

        # Select speaker
        speaker = SARVAM_SPEAKERS.get(lang_code, "ritu")

        payload = {
            "inputs": [text],
            "target_language_code": lang_code,
            "speaker": speaker,
            "pitch": 0,
            "pace": 1.08,
            "loudness": 1.5,
            "speech_sample_rate": 16000,
            "enable_preprocessing": True,
            "model": "bulbul:v3",
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()

        audios = result.get("audios", [])
        if not audios:
            raise ValueError("No audio returned by Sarvam TTS")

        wav_bytes = base64.b64decode(audios[0])
        return wav_bytes, "audio/wav"

    async def _synthesize_edge_tts(
        self, text: str, lang_code: str
    ) -> Tuple[bytes, str]:
        """Synthesize using Microsoft Edge-TTS neural voices (Free, fast, low latency)."""
        voice = EDGE_TTS_VOICES.get(lang_code, EDGE_TTS_VOICES["en-IN"])
        communicate = edge_tts.Communicate(text=text, voice=voice, rate="+8%")

        mp3_io = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_io.write(chunk["data"])

        return mp3_io.getvalue(), "audio/mpeg"
