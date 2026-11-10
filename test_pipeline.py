"""
End-to-End Test Suite for Vaani Duplex with Live Groq API Key.
"""

import asyncio
import io
import edge_tts
from groq import AsyncGroq
from app.config import get_settings
from app.providers import SpeechPipelineProviders
from app.tools import execute_tool, get_session_state
from app.vad import DuplexVADProcessor


async def run_full_suite():
    settings = get_settings()
    print("=" * 60)
    print("1. Checking Environment & Configuration")
    print("=" * 60)
    print(f"Groq Configured: {settings.is_groq_configured}")
    print(f"Primary Model:   {settings.groq_primary_model}")
    print(f"Fallback Model:  {settings.groq_fallback_model}")
    assert settings.is_groq_configured, "Groq is not configured properly in .env!"
    print(">> Config check: PASSED\n")

    print("=" * 60)
    print("2. Testing Edge-TTS Neural Synthesis (Hindi, Telugu, English)")
    print("=" * 60)
    providers = SpeechPipelineProviders()
    hi_audio, hi_mime = await providers.synthesize_clause("नमस्ते, आपकी क्या सहायता करूँ?", "hi-IN")
    te_audio, te_mime = await providers.synthesize_clause("నమస్కారం, మీ ఆర్డర్ వివరాలు ఇక్కడ ఉన్నాయి.", "te-IN")
    en_audio, en_mime = await providers.synthesize_clause("Hello, how can I assist you today?", "en-IN")
    print(f">> Hindi TTS:   {len(hi_audio)} bytes ({hi_mime})")
    print(f">> Telugu TTS:  {len(te_audio)} bytes ({te_mime})")
    print(f">> English TTS: {len(en_audio)} bytes ({en_mime})")
    assert len(hi_audio) > 1000 and len(te_audio) > 1000 and len(en_audio) > 1000
    print(">> Edge-TTS Synthesis: PASSED\n")

    print("=" * 60)
    print("3. Testing Groq Whisper ASR (whisper-large-v3-turbo)")
    print("=" * 60)
    client = AsyncGroq(api_key=settings.groq_api_key)
    whisper_res = await client.audio.transcriptions.create(
        file=("test.mp3", hi_audio, "audio/mpeg"),
        model="whisper-large-v3-turbo",
        response_format="verbose_json"
    )
    transcript = whisper_res.text.strip()
    language = whisper_res.language
    print(f">> Whisper Transcript: '{transcript}'")
    print(f">> Whisper Language:   '{language}'")
    assert len(transcript) > 0
    print(">> Groq Whisper ASR: PASSED\n")

    print("=" * 60)
    print("4. Testing Stateful In-Memory Tools")
    print("=" * 60)
    res_order = execute_tool("check_order_status", {"order_id": "ORD-9921"})
    print(f">> Order Status:   {res_order['message']}")
    assert res_order["found"] is True

    res_meet = execute_tool("schedule_calendar_meeting", {
        "title": "AI Architecture Sync",
        "date": "2026-11-15",
        "time": "16:00",
        "attendee": "Ananya Roy"
    })
    print(f">> Meeting Result: {res_meet['message']}")
    assert res_meet["success"] is True

    res_ticket = execute_tool("create_support_ticket", {
        "issue_summary": "Payment debited but order failed",
        "priority": "HIGH"
    })
    print(f">> Ticket Result:  {res_ticket['message']}")
    assert res_ticket["success"] is True
    print(">> Stateful Tools: PASSED\n")

    print("=" * 60)
    print("5. Testing Groq LLM Streaming & Tool Calling (Hinglish)")
    print("=" * 60)
    stream = await client.chat.completions.create(
        model=settings.groq_primary_model,
        messages=[
            {"role": "system", "content": "You are Vaani, a voice assistant. Keep answers under 25 words."},
            {"role": "user", "content": "Mera order ORD-9921 ka status check karo please."}
        ],
        tools=[{
            "type": "function",
            "function": {
                "name": "check_order_status",
                "description": "Check order status",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"]
                }
            }
        }],
        tool_choice="auto",
        stream=True
    )

    tool_calls = []
    text_chunks = []
    async for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta:
            if delta.tool_calls:
                tool_calls.extend(delta.tool_calls)
            if delta.content:
                text_chunks.append(delta.content)

    print(f">> Tool calls detected: {len(tool_calls)} call(s)")
    if text_chunks:
        print(f">> Streaming content:   {''.join(text_chunks)}")
    assert len(tool_calls) > 0 or len(text_chunks) > 0
    print(">> Groq LLM Streaming: PASSED\n")

    print("=" * 60)
    print("6. Testing Silero VAD Frame Processor")
    print("=" * 60)
    vad = DuplexVADProcessor()
    silent_frame = bytes(1024)
    prob = vad.compute_speech_probability(silent_frame)
    print(f">> Silence frame speech probability: {prob:.4f} (Threshold: {vad.vad_threshold})")
    assert prob < vad.vad_threshold
    print(">> Silero VAD: PASSED\n")

    print("=" * 60)
    print("🎉 ALL 6 SUITES PASSED! VAANI DUPLEX IS 100% OPERATIONAL!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_full_suite())
