"""
Voice Session Orchestrator for Vaani Duplex.
Implements:
1. Multilingual Prompt Mirroring (Hinglish, Telugish, Indian English)
2. Speculative Clause-Boundary Streaming with Groq (70B -> 8B fallback)
3. Sub-800ms TTFA telemetry dispatch
4. Stateful tool execution and spoken confirmations
5. Real-time barge-in interruption (<15ms client clearance)
"""

import asyncio
import base64
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from fastapi import WebSocket
from groq import AsyncGroq

from app.config import get_settings
from app.providers import SpeechPipelineProviders
from app.tools import TOOL_SCHEMAS, execute_tool, get_session_state
from app.vad import DuplexVADProcessor

logger = logging.getLogger("vaani.pipeline")

# Speculative clause splitter: at least 24 chars ending with punctuation
CLAUSE_REGEX = re.compile(r"(.{24,}?[।.?!,;:])(\s+|$)")

SYSTEM_PROMPT = """You are Vaani (वाणी / వాణి), an ultra-fast, conversational Indian voice AI assistant.
Your capabilities and persona:
1. MULTILINGUAL & CODE-SWITCHING: You natively understand and speak Hindi, Telugu, and Indian English.
   - You dynamically mirror the user's language and dialect.
   - If the user speaks Hinglish (e.g., 'Mera order status check karo please'), reply naturally in warm, polite Hinglish.
   - If the user speaks Telugish (e.g., 'Naa order ORD-9921 status emititi?'), reply naturally in authentic Telugish.
   - If the user speaks Indian English, reply in natural conversational Indian English.
2. SPOKEN CONCISION:
   - Keep answers strictly under 35 words (1 to 2 crisp, spoken sentences).
   - Talk like a real phone assistant: warm, helpful, direct.
   - NEVER use markdown, bolding, asterisks, bullet points, numbered lists, emojis, or code blocks. Everything you output goes directly to a text-to-speech voice synthesizer.
3. STATEFUL TOOL CALLING:
   - Proactively call tools when the user wants to schedule an appointment, create a support ticket, or check an order status (sample IDs: ORD-9921 or ORD-4410).
   - Once a tool completes, speak a 1-sentence confirmation mirroring the user's language.
"""


class VoiceSessionOrchestrator:
    """
    Manages the lifecycle of a full-duplex voice conversation over WebSocket.
    """

    def __init__(
        self,
        websocket: WebSocket,
        vad_processor: DuplexVADProcessor,
        providers: SpeechPipelineProviders,
    ) -> None:
        self.ws = websocket
        self.vad = vad_processor
        self.providers = providers
        self.settings = get_settings()

        self.active_provider: str = self.settings.speech_provider
        self.conversation_history: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        self._barge_in_event = asyncio.Event()
        self._current_turn_task: Optional[asyncio.Task] = None
        self._is_agent_speaking = False

    def set_provider(self, provider: str) -> None:
        """Switch speech provider between 'free' and 'sarvam'."""
        if provider in ("free", "sarvam"):
            self.active_provider = provider
            logger.info("Active speech provider switched to: %s", provider)

    def trigger_barge_in(self) -> None:
        """
        Interrupts agent speech immediately. Cancels generation tasks, resets VAD,
        and tells the browser client to flush audio playback queue immediately.
        """
        logger.info("⚡ Executing Barge-In: Cancelling turn and clearing client playback.")
        self._barge_in_event.set()
        self._is_agent_speaking = False
        self.vad.set_agent_speaking(False)

        # Cancel running LLM/TTS task if active
        if self._current_turn_task and not self._current_turn_task.done():
            self._current_turn_task.cancel()

        # Send control signal to client to stop audio playback in <15ms
        asyncio.create_task(
            self._safe_send_json(
                {
                    "type": "control",
                    "action": "clear_playback",
                    "reason": "barge_in",
                    "timestamp": time.time(),
                }
            )
        )

    def on_playback_finished(self) -> None:
        """Called when browser finishes playing queued audio chunks."""
        self._is_agent_speaking = False
        self.vad.set_agent_speaking(False)

    async def _safe_send_json(self, payload: Dict[str, Any]) -> None:
        """Safely dispatches JSON message over WebSocket."""
        try:
            await self.ws.send_text(json.dumps(payload))
        except Exception as e:
            logger.debug("WebSocket send error: %s", e)

    async def handle_user_turn(self, pcm_bytes: bytes) -> None:
        """
        Processes a completed user voice turn:
        1. Transcribe audio (ASR)
        2. Detect language
        3. Stream LLM tokens with speculative clause synthesis (TTS)
        4. Execute tools if required
        """
        # Cancel any previous task still lingering
        if self._current_turn_task and not self._current_turn_task.done():
            self._current_turn_task.cancel()

        self._current_turn_task = asyncio.create_task(self._process_turn_lifecycle(pcm_bytes))

    async def _process_turn_lifecycle(self, pcm_bytes: bytes) -> None:
        turn_start_time = time.perf_counter()
        self._barge_in_event.clear()

        # -------------------------------------------------------------
        # STEP 1: ASR Transcription
        # -------------------------------------------------------------
        asr_start = time.perf_counter()
        try:
            transcript, detected_lang = await self.providers.transcribe(
                pcm_bytes, provider=self.active_provider
            )
        except Exception as e:
            logger.error("ASR transcription failed: %s", e)
            await self._safe_send_json(
                {
                    "type": "error",
                    "message": f"Transcription error: {str(e)}",
                }
            )
            return

        asr_ms = int((time.perf_counter() - asr_start) * 1000)

        if not transcript or len(transcript.strip()) == 0:
            logger.debug("Empty transcription, ignoring turn.")
            return

        logger.info(
            "User said: '%s' [lang=%s, asr=%dms]", transcript, detected_lang, asr_ms
        )

        # Notify client of user transcript and language
        await self._safe_send_json(
            {
                "type": "transcript",
                "role": "user",
                "text": transcript,
                "lang": detected_lang,
            }
        )

        # Emit initial telemetry card update
        await self._safe_send_json(
            {
                "type": "telemetry",
                "asr_ms": asr_ms,
                "ttft_ms": None,
                "ttfa_ms": None,
                "lang": detected_lang,
            }
        )

        # -------------------------------------------------------------
        # STEP 2: Agent Turn - Speculative Clause Pipeline
        # -------------------------------------------------------------
        self._is_agent_speaking = True
        self.vad.set_agent_speaking(True)

        self.conversation_history.append({"role": "user", "content": transcript})

        await self._stream_assistant_response(
            detected_lang=detected_lang,
            turn_start_time=turn_start_time,
            asr_ms=asr_ms,
        )

    async def _stream_assistant_response(
        self,
        detected_lang: str,
        turn_start_time: float,
        asr_ms: int,
        is_tool_followup: bool = False,
    ) -> None:
        """Streams LLM tokens, detects tool calls, and dispatches clauses to TTS."""
        tts_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        tts_worker_task = asyncio.create_task(
            self._tts_worker(
                tts_queue=tts_queue,
                lang_code=detected_lang,
                turn_start_time=turn_start_time,
                asr_ms=asr_ms,
            )
        )

        groq_key = self.settings.groq_api_key.strip()
        client = AsyncGroq(api_key=groq_key)

        model_to_use = self.settings.groq_primary_model
        llm_start = time.perf_counter()
        ttft_recorded = False
        ttft_ms = 0

        buffer = ""
        accumulated_assistant_text = ""
        tool_calls_accumulator: Dict[int, Dict[str, Any]] = {}

        # Candidate models with resilient fallback chain
        candidate_models = []
        for m in [self.settings.groq_primary_model, self.settings.groq_fallback_model, "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]:
            if m and m not in candidate_models:
                candidate_models.append(m)

        stream = None
        for candidate in candidate_models:
            try:
                stream = await self._call_groq_stream(client, candidate)
                model_to_use = candidate
                break
            except Exception as e:
                logger.warning("Groq model %s failed (%s), trying next fallback...", candidate, e)

        if not stream:
            logger.error("All Groq LLM candidate models failed.")
            await tts_queue.put(None)
            await tts_worker_task
            return

        try:
            async for chunk in stream:
                if self._barge_in_event.is_set():
                    logger.info("Barge-in detected during LLM streaming. Aborting stream.")
                    break

                delta = chunk.choices[0].delta if chunk.choices else None
                if not delta:
                    continue

                # Record TTFT on first token
                if not ttft_recorded:
                    ttft_ms = int((time.perf_counter() - llm_start) * 1000)
                    ttft_recorded = True
                    await self._safe_send_json(
                        {
                            "type": "telemetry",
                            "asr_ms": asr_ms,
                            "ttft_ms": ttft_ms,
                            "ttfa_ms": None,
                            "lang": detected_lang,
                        }
                    )

                # Collect tool call deltas
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_accumulator:
                            tool_calls_accumulator[idx] = {
                                "id": tc.id or f"call_{idx}",
                                "name": tc.function.name if tc.function and tc.function.name else "",
                                "arguments": "",
                            }
                        if tc.function:
                            if tc.function.name:
                                tool_calls_accumulator[idx]["name"] = tc.function.name
                            if tc.function.arguments:
                                tool_calls_accumulator[idx]["arguments"] += tc.function.arguments

                # Collect text deltas and check for speculative clause boundaries
                if delta.content:
                    token_text = delta.content
                    buffer += token_text
                    accumulated_assistant_text += token_text

                    # Regex match: at least 24 chars ending in punctuation
                    match = CLAUSE_REGEX.search(buffer)
                    if match:
                        clause = match.group(1).strip()
                        buffer = buffer[match.end():]
                        if clause:
                            await tts_queue.put(clause)

            # Flush any remaining buffer text
            if buffer.strip() and not self._barge_in_event.is_set():
                await tts_queue.put(buffer.strip())

        except asyncio.CancelledError:
            logger.info("LLM stream task cancelled.")
            await tts_queue.put(None)
            await tts_worker_task
            raise
        except Exception as e:
            logger.error("LLM streaming exception: %s", e)
        finally:
            # Signal end of queue to TTS worker
            await tts_queue.put(None)
            await tts_worker_task

        # Record assistant reply in history
        if accumulated_assistant_text:
            self.conversation_history.append(
                {"role": "assistant", "content": accumulated_assistant_text}
            )
            await self._safe_send_json(
                {
                    "type": "transcript",
                    "role": "assistant",
                    "text": accumulated_assistant_text,
                    "lang": detected_lang,
                }
            )

        # -------------------------------------------------------------
        # STEP 3: Handle Tool Execution (if any tool was requested)
        # -------------------------------------------------------------
        if tool_calls_accumulator and not self._barge_in_event.is_set():
            logger.info("Executing %d tool call(s)...", len(tool_calls_accumulator))
            tool_messages_to_add = []
            tool_calls_list = []

            for _, tc in tool_calls_accumulator.items():
                name = tc["name"]
                args_str = tc["arguments"]
                call_id = tc["id"]

                tool_calls_list.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": args_str},
                    }
                )

                # Execute in-memory tool
                tool_result = execute_tool(name, args_str)
                logger.info("Tool %s executed: %s", name, tool_result)

                # Dispatch tool call event to web UI
                await self._safe_send_json(
                    {
                        "type": "tool_call",
                        "name": name,
                        "args": args_str,
                        "result": tool_result,
                        "session_state": get_session_state(),
                    }
                )

                tool_messages_to_add.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": json.dumps(tool_result),
                    }
                )

            # Append assistant's tool call intent and the tool results
            self.conversation_history.append(
                {
                    "role": "assistant",
                    "content": accumulated_assistant_text or None,
                    "tool_calls": tool_calls_list,
                }
            )
            self.conversation_history.extend(tool_messages_to_add)

            # Stream spoken confirmation to user following tool execution
            await self._stream_assistant_response(
                detected_lang=detected_lang,
                turn_start_time=turn_start_time,
                asr_ms=asr_ms,
                is_tool_followup=True,
            )

    async def _call_groq_stream(self, client: AsyncGroq, model: str):
        """Dispatches streaming completion to Groq."""
        return await client.chat.completions.create(
            model=model,
            messages=self.conversation_history,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.4,
            max_tokens=150,
            stream=True,
        )

    async def _tts_worker(
        self,
        tts_queue: asyncio.Queue[Optional[str]],
        lang_code: str,
        turn_start_time: float,
        asr_ms: int,
    ) -> None:
        """
        Asynchronously synthesizes clauses from the queue and sends audio chunks
        over WebSocket to the browser player.
        """
        clause_index = 0
        first_audio_sent = False

        while True:
            try:
                clause = await tts_queue.get()
                if clause is None:
                    # End of turn stream
                    break

                if self._barge_in_event.is_set():
                    logger.debug("Barge-in active, skipping TTS clause.")
                    continue

                clause_synth_start = time.perf_counter()
                audio_bytes, mime_type = await self.providers.synthesize_clause(
                    clause, lang_code, provider=self.active_provider
                )

                if self._barge_in_event.is_set() or not audio_bytes:
                    continue

                clause_index += 1
                audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

                # Measure TTFA on the very first synthesized clause
                if not first_audio_sent:
                    ttfa_ms = int((time.perf_counter() - turn_start_time) * 1000)
                    first_audio_sent = True
                    logger.info(
                        "🚀 First audio clause synthesized! TTFA = %dms [ASR: %dms, Synth: %dms]",
                        ttfa_ms,
                        asr_ms,
                        int((time.perf_counter() - clause_synth_start) * 1000),
                    )
                    await self._safe_send_json(
                        {
                            "type": "telemetry",
                            "asr_ms": asr_ms,
                            "ttft_ms": None,
                            "ttfa_ms": ttfa_ms,
                            "lang": lang_code,
                        }
                    )

                # Send chunk to client
                await self._safe_send_json(
                    {
                        "type": "audio",
                        "clause": clause,
                        "audio_base64": audio_b64,
                        "mime_type": mime_type,
                        "clause_index": clause_index,
                    }
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("TTS worker exception: %s", e)
