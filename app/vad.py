"""
Silero VAD processor for real-time Full-Duplex audio streaming.
Implements a 3-state machine for:
1. Backchannel Ignore (< 192ms sound ignored during agent speech)
2. True Barge-In (>= 192ms speech interrupts agent immediately)
3. Normal Turn-Taking (SPEECH_START at 96ms, SPEECH_END at 576ms silence)
"""

import collections
import logging
from typing import Optional, TypedDict
import numpy as np
import torch

from app.config import get_settings

logger = logging.getLogger("vaani.vad")


class VADEvent(TypedDict):
    event: str  # "SPEECH_START", "SPEECH_CONTINUE", "SPEECH_END", "BARGE_IN", "BACKCHANNEL_IGNORE"
    prob: float
    audio: Optional[bytes]


class DuplexVADProcessor:
    """
    Real-time 16kHz Int16 PCM VAD processor with stateful barge-in and
    backchannel discernment.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.sample_rate = self.settings.sample_rate
        self.frame_samples = self.settings.frame_samples
        self.frame_bytes = self.settings.frame_bytes
        self.vad_threshold = self.settings.vad_threshold
        self.barge_in_min_frames = self.settings.barge_in_min_frames
        self.silence_end_frames = self.settings.silence_end_frames
        self.speech_start_min_frames = self.settings.speech_start_min_frames

        # Ring buffer for pre-roll (5 frames = 160ms) to preserve onset consonants
        self.pre_roll_buffer: collections.deque[bytes] = collections.deque(
            maxlen=self.settings.pre_roll_frames
        )

        # Buffer for active user speech
        self.audio_buffer = bytearray()

        # State tracking
        self.agent_is_speaking: bool = False
        self.is_buffering_utterance: bool = False
        self.barge_in_triggered: bool = False

        self.consecutive_voiced: int = 0
        self.consecutive_silence: int = 0

        # Load Silero VAD model (prefer ONNX, fallback to Torch)
        self.model = self._load_model()

    def _load_model(self):
        """Loads Silero VAD using ONNX runtime if available, else torch."""
        try:
            from silero_vad import load_silero_vad
            try:
                model = load_silero_vad(onnx=True)
                logger.info("Silero VAD initialized successfully using ONNX backend.")
                return model
            except Exception as e_onnx:
                logger.warning(
                    "ONNX Silero VAD load failed (%s), trying Torch backend...", e_onnx
                )
                model = load_silero_vad(onnx=False)
                logger.info("Silero VAD initialized successfully using Torch backend.")
                return model
        except Exception as e:
            logger.error("Failed to load silero_vad package: %s", e)
            return None

    def reset_states(self) -> None:
        """Resets Silero model internal recurrent states if available."""
        if self.model and hasattr(self.model, "reset_states"):
            try:
                self.model.reset_states()
            except Exception:
                pass
        self.consecutive_voiced = 0
        self.consecutive_silence = 0
        self.is_buffering_utterance = False
        self.barge_in_triggered = False
        self.audio_buffer.clear()
        self.pre_roll_buffer.clear()

    def set_agent_speaking(self, is_speaking: bool) -> None:
        """Notifies the VAD whether the agent is currently synthesizing/playing audio."""
        self.agent_is_speaking = is_speaking
        if not is_speaking:
            self.barge_in_triggered = False
            self.consecutive_voiced = 0

    def compute_speech_probability(self, frame_bytes: bytes) -> float:
        """
        Calculates speech probability for a 512-sample (1024-byte) Int16 PCM frame.
        """
        if len(frame_bytes) != self.frame_bytes:
            return 0.0

        int16_data = np.frombuffer(frame_bytes, dtype=np.int16)
        float32_data = int16_data.astype(np.float32) / 32768.0

        if self.model is not None:
            try:
                tensor = torch.from_numpy(float32_data)
                # Silero expects (samples,) or (1, samples)
                prob = self.model(tensor, self.sample_rate).item()
                return float(prob)
            except Exception as e:
                logger.debug("Silero inference error: %s", e)

        # Fallback energy-based VAD if model is unavailable
        energy = float(np.sqrt(np.mean(float32_data**2)))
        return 0.9 if energy > 0.02 else 0.1

    def process_frame(self, frame_bytes: bytes) -> Optional[VADEvent]:
        """
        Processes a single 32ms (1024-byte) PCM frame through the 3-state machine.
        Returns a VADEvent or None.
        """
        if len(frame_bytes) != self.frame_bytes:
            return None

        prob = self.compute_speech_probability(frame_bytes)
        is_voiced = prob >= self.vad_threshold

        # Always maintain the 5-frame pre-roll ring buffer
        self.pre_roll_buffer.append(frame_bytes)

        # =====================================================================
        # STATE 1 & 2: Agent is currently speaking (Full-Duplex Barge-In vs Backchannel)
        # =====================================================================
        if self.agent_is_speaking:
            if is_voiced:
                self.consecutive_voiced += 1
                self.consecutive_silence = 0

                # True Barge-In: Voiced frames reach BARGE_IN_MIN_FRAMES (>= 6 frames / 192ms)
                if self.consecutive_voiced >= self.barge_in_min_frames:
                    if not self.barge_in_triggered:
                        self.barge_in_triggered = True
                        self.is_buffering_utterance = True
                        self.audio_buffer.clear()
                        # Pre-roll buffered frames to preserve leading consonants
                        for pre_frame in self.pre_roll_buffer:
                            self.audio_buffer.extend(pre_frame)

                        logger.info(
                            "⚡ Barge-in triggered after %d voiced frames (prob=%.2f)",
                            self.consecutive_voiced,
                            prob,
                        )
                        return {
                            "event": "BARGE_IN",
                            "prob": prob,
                            "audio": None,
                        }
                    else:
                        self.audio_buffer.extend(frame_bytes)
                        return {
                            "event": "SPEECH_CONTINUE",
                            "prob": prob,
                            "audio": None,
                        }
                else:
                    # Voiced sound is < 192ms: treat as harmless backchannel ("haan", "hmm", "avunu")
                    return {
                        "event": "BACKCHANNEL_IGNORE",
                        "prob": prob,
                        "audio": None,
                    }
            else:
                # Silence while agent is speaking
                if self.barge_in_triggered:
                    self.consecutive_silence += 1
                    self.audio_buffer.extend(frame_bytes)

                    # After barge-in, check if user has now stopped speaking
                    if self.consecutive_silence >= self.silence_end_frames:
                        complete_audio = bytes(self.audio_buffer)
                        self.barge_in_triggered = False
                        self.is_buffering_utterance = False
                        self.consecutive_voiced = 0
                        self.consecutive_silence = 0
                        self.audio_buffer.clear()
                        logger.info("User utterance completed post-barge-in (%d bytes)", len(complete_audio))
                        return {
                            "event": "SPEECH_END",
                            "prob": prob,
                            "audio": complete_audio,
                        }
                    return {
                        "event": "SPEECH_CONTINUE",
                        "prob": prob,
                        "audio": None,
                    }
                else:
                    # User was silent or finished a short backchannel chirp without barging in
                    self.consecutive_voiced = 0
                    self.consecutive_silence += 1
                    return None

        # =====================================================================
        # STATE 3: Normal Turn-Taking (Agent is silent, waiting for or listening to user)
        # =====================================================================
        if is_voiced:
            self.consecutive_voiced += 1
            self.consecutive_silence = 0

            if not self.is_buffering_utterance:
                # Start speech turn after speech_start_min_frames (3 frames / 96ms)
                if self.consecutive_voiced >= self.speech_start_min_frames:
                    self.is_buffering_utterance = True
                    self.audio_buffer.clear()
                    for pre_frame in self.pre_roll_buffer:
                        self.audio_buffer.extend(pre_frame)

                    return {
                        "event": "SPEECH_START",
                        "prob": prob,
                        "audio": None,
                    }
                return None
            else:
                self.audio_buffer.extend(frame_bytes)
                return {
                    "event": "SPEECH_CONTINUE",
                    "prob": prob,
                    "audio": None,
                }
        else:
            if self.is_buffering_utterance:
                self.consecutive_silence += 1
                self.audio_buffer.extend(frame_bytes)

                # Silence threshold reached (18 frames / 576ms) -> dispatch complete audio
                if self.consecutive_silence >= self.silence_end_frames:
                    complete_audio = bytes(self.audio_buffer)
                    self.is_buffering_utterance = False
                    self.consecutive_voiced = 0
                    self.consecutive_silence = 0
                    self.audio_buffer.clear()
                    logger.info("Speech turn ended normally (%d bytes)", len(complete_audio))
                    return {
                        "event": "SPEECH_END",
                        "prob": prob,
                        "audio": complete_audio,
                    }
                return {
                    "event": "SPEECH_CONTINUE",
                    "prob": prob,
                    "audio": None,
                }
            else:
                self.consecutive_voiced = 0
                self.consecutive_silence += 1
                return None
