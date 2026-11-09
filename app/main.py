"""
FastAPI Server and WebSocket Endpoint for Vaani Duplex.
Handles:
1. Static web asset serving (HTML/JS/CSS)
2. State inspection and reset endpoints
3. Full-duplex binary PCM WebSocket streaming with frame-level VAD
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.pipeline import VoiceSessionOrchestrator
from app.providers import SpeechPipelineProviders
from app.tools import get_session_state, reset_session_state
from app.vad import DuplexVADProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("vaani.server")

settings = get_settings()
providers_singleton = SpeechPipelineProviders()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("  🚀 Starting Vaani Duplex Voice Agent Server")
    logger.info("  - Speech Provider Default: %s", settings.speech_provider)
    logger.info("  - Groq Configured: %s", settings.is_groq_configured)
    logger.info("  - Sarvam Configured: %s", settings.is_sarvam_configured)
    logger.info("  - Sample Rate: %d Hz (Frame: %d bytes)", settings.sample_rate, settings.frame_bytes)
    logger.info("=" * 60)
    yield
    logger.info("Shutting down Vaani Duplex.")


app = FastAPI(
    title="Vaani Duplex Voice Agent",
    description="Full-Duplex Multilingual Indian Voice Agent (Hindi, Telugu, Indian English)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static folder exists
static_path = Path(__file__).resolve().parent.parent / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


@app.get("/")
async def get_index():
    """Serves the developer console web application."""
    index_file = static_path / "index.html"
    if not index_file.exists():
        return JSONResponse({"error": "static/index.html not found"}, status_code=404)
    return FileResponse(str(index_file))


@app.get("/api/state")
async def get_current_state():
    """Returns current in-memory database and configuration status."""
    return JSONResponse(
        {
            "session_state": get_session_state(),
            "config": {
                "default_provider": settings.speech_provider,
                "groq_configured": settings.is_groq_configured,
                "sarvam_configured": settings.is_sarvam_configured,
                "primary_model": settings.groq_primary_model,
                "fallback_model": settings.groq_fallback_model,
                "vad_threshold": settings.vad_threshold,
                "barge_in_min_frames": settings.barge_in_min_frames,
                "silence_end_frames": settings.silence_end_frames,
            },
        }
    )


@app.post("/api/state/reset")
async def reset_state():
    """Resets appointments, tickets, and order demo state."""
    fresh_state = reset_session_state()
    return JSONResponse({"status": "reset_success", "session_state": fresh_state})


@app.websocket("/ws/voice")
async def voice_websocket_endpoint(websocket: WebSocket):
    """
    Bidirectional WebSocket connection for full-duplex voice communication.
    Accepts:
      - Binary: 16kHz 16-bit Mono Int16 PCM chunks from user's microphone.
      - Text/JSON: Session controls ('set_provider', 'playback_finished', 'text_turn').
    Emits:
      - Binary/JSON: Transcripts, speculative audio clauses, telemetry, tool updates, barge-in signals.
    """
    await websocket.accept()
    logger.info("Client connected to /ws/voice")

    vad_processor = DuplexVADProcessor()
    orchestrator = VoiceSessionOrchestrator(
        websocket=websocket,
        vad_processor=vad_processor,
        providers=providers_singleton,
    )

    # Buffer for aggregating fractional PCM chunks into exact 1024-byte VAD frames
    pcm_slice_buffer = bytearray()
    frame_bytes = settings.frame_bytes  # 1024 bytes (512 samples)

    # Send initial welcome and configuration status
    await websocket.send_text(
        json.dumps(
            {
                "type": "ready",
                "message": "Vaani Duplex WebSocket Ready",
                "groq_ready": settings.is_groq_configured,
                "sarvam_ready": settings.is_sarvam_configured,
                "active_provider": orchestrator.active_provider,
                "session_state": get_session_state(),
            }
        )
    )

    try:
        while True:
            message = await websocket.receive()

            # ---------------------------------------------------------
            # 1. Binary message: Raw PCM audio from browser microphone
            # ---------------------------------------------------------
            if "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]
                pcm_slice_buffer.extend(raw_bytes)

                # Process all complete 1024-byte frames in the buffer
                while len(pcm_slice_buffer) >= frame_bytes:
                    frame = bytes(pcm_slice_buffer[:frame_bytes])
                    del pcm_slice_buffer[:frame_bytes]

                    vad_event = vad_processor.process_frame(frame)
                    if not vad_event:
                        continue

                    event_name = vad_event["event"]

                    if event_name == "BARGE_IN":
                        # Immediate interruption of agent
                        orchestrator.trigger_barge_in()
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "event",
                                    "name": "BARGE_IN",
                                    "prob": round(vad_event["prob"], 2),
                                    "detail": "Interrupted agent speech",
                                }
                            )
                        )

                    elif event_name == "SPEECH_START":
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "event",
                                    "name": "SPEECH_START",
                                    "prob": round(vad_event["prob"], 2),
                                }
                            )
                        )

                    elif event_name == "SPEECH_END":
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "event",
                                    "name": "SPEECH_END",
                                    "prob": round(vad_event["prob"], 2),
                                }
                            )
                        )

                        # User turn completed -> dispatch to pipeline
                        user_audio = vad_event.get("audio")
                        if user_audio and len(user_audio) > (frame_bytes * 3):
                            asyncio.create_task(
                                orchestrator.handle_user_turn(user_audio)
                            )

            # ---------------------------------------------------------
            # 2. Text/JSON message: Client controls & signals
            # ---------------------------------------------------------
            elif "text" in message and message["text"]:
                try:
                    data = json.loads(message["text"])
                    msg_type = data.get("type", "")

                    if msg_type == "playback_finished":
                        orchestrator.on_playback_finished()

                    elif msg_type == "set_provider":
                        new_prov = data.get("provider", "free")
                        orchestrator.set_provider(new_prov)
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "provider_changed",
                                    "active_provider": orchestrator.active_provider,
                                }
                            )
                        )

                    elif msg_type == "text_input":
                        # Support direct text testing (e.g. from developer console)
                        query_text = data.get("text", "").strip()
                        if query_text:
                            orchestrator.conversation_history.append(
                                {"role": "user", "content": query_text}
                            )
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "transcript",
                                        "role": "user",
                                        "text": query_text,
                                        "lang": "en-IN",
                                    }
                                )
                            )
                            # Stream response
                            asyncio.create_task(
                                orchestrator._stream_assistant_response(
                                    detected_lang="en-IN",
                                    turn_start_time=asyncio.get_event_loop().time(),
                                    asr_ms=0,
                                )
                            )

                    elif msg_type == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))

                except Exception as e:
                    logger.debug("Failed parsing text message: %s", e)

    except WebSocketDisconnect:
        logger.info("Client disconnected from /ws/voice")
    except Exception as e:
        logger.error("Unhandled error in /ws/voice loop: %s", e)
    finally:
        orchestrator.trigger_barge_in()
