# Vaani Duplex — Setup & Testing Guide

This guide walks you through setting up and running **Vaani Duplex** from scratch, configuring your free API key, and performing live voice tests with stateful tool calling and barge-in interruptions.

---

## 1. Quick Environment Setup

### Step 1.1: Create & Activate Virtual Environment
Open PowerShell or your terminal in the project directory (`c:\Code\Voice AI`):

```powershell
# Create a virtual environment named .venv
python -m venv .venv

# Activate the virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
```

*(If PowerShell script execution is restricted on your machine, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

### Step 1.2: Install Required Dependencies
Install the required packages:

```powershell
pip install -r requirements.txt
```

---

## 2. API Key Configuration

The project operates **100% free (₹0 cost)** using Groq.

1. Open the file `.env` in the root of the project directory.
2. Locate the line:
   ```env
   GROQ_API_KEY=gsk_paste_your_free_groq_key_here
   ```
3. Replace `gsk_paste_your_free_groq_key_here` with your free Groq API key (get one for free in 30 seconds at [https://console.groq.com/keys](https://console.groq.com/keys)).
4. *(Optional)* If you have a Sarvam AI API key, you can paste it into:
   ```env
   SARVAM_API_KEY=
   ```
   If left blank, the application automatically defaults to Free Mode (`SPEECH_PROVIDER=free`).

Your `.env` file should look like this:
```env
GROQ_API_KEY=gsk_your_actual_groq_key_here
SARVAM_API_KEY=
SPEECH_PROVIDER=free
GROQ_PRIMARY_MODEL=llama-3.3-70b-versatile
GROQ_FALLBACK_MODEL=llama-3.1-8b-instant
VAD_THRESHOLD=0.65
BARGE_IN_MIN_FRAMES=6
SILENCE_END_FRAMES=18
```

---

## 3. Launching the Voice Agent

Start the FastAPI application using `uvicorn`:

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Once running, you will see output similar to:
```
INFO:     [vaani.server] ============================================================
INFO:     [vaani.server]   🚀 Starting Vaani Duplex Voice Agent Server
INFO:     [vaani.server]   - Speech Provider Default: free
INFO:     [vaani.server]   - Groq Configured: True
INFO:     [vaani.server]   - Sarvam Configured: False
INFO:     [vaani.server]   - Sample Rate: 16000 Hz (Frame: 1024 bytes)
INFO:     [vaani.server] ============================================================
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Open your browser and navigate to:
👉 **[http://localhost:8000](http://localhost:8000)**

---

## 4. Live Voice Interaction & Testing Checklist

Allow microphone access when prompted by your browser. Click **"Start Microphone"** to open full-duplex communication.

### Test Script 1: Hinglish Meeting Scheduling (Tool Calling)
- **Goal:** Verify that Hinglish code-switching is naturally understood and triggers the `schedule_calendar_meeting` tool.
- **What to say into your mic:**
  > *"Vaani, Rohan Sharma ke saath kal dopahar 2 baje quarterly review meeting schedule karo please."*
- **Expected Behavior:**
  1. ASR will transcribe the speech and classify language as `hi-IN` (Hinglish).
  2. The console displays a yellow pill: `🛠️ Tool Executed: schedule_calendar_meeting`.
  3. The **Appointments tab** on the right side immediately updates with the new appointment record (`APT-xxx`).
  4. Vaani speaks a polite, natural Hinglish confirmation:
     > *"Theek hai, Rohan Sharma ke saath kal dopahar do baje meeting schedule ho gayi hai."*
  5. The End-to-End TTFA telemetry card reads under **800ms**.

---

### Test Script 2: Telugish Order Tracking (Tool Calling)
- **Goal:** Verify Telugish code-switching, order lookup (`ORD-9921`), and Telugu speech synthesis.
- **What to say into your mic:**
  > *"Naa order ORD-9921 status emititi? Delivery eppudu vastundi?"*
- **Expected Behavior:**
  1. ASR transcribes the sentence and identifies language as `te-IN` (Telugu/Telugish).
  2. The console displays: `🛠️ Tool Executed: check_order_status`.
  3. The **Orders tab** highlights the details for `ORD-9921` (Sony WH-1000XM5, Out for Delivery, Bluedart).
  4. Vaani responds in authentic Telugish:
     > *"Mee order ORD-9921 out for delivery undi, ee roju saayantram 6:00 PM lopu deliver avtundi."*
  5. The speech is synthesized using `te-IN-ShrutiNeural` (or Sarvam `kavya` if enabled).

---

### Test Script 3: Barge-In Interruption & Backchannel Test
- **Goal:** Verify that short verbal nods (<192ms) do not interrupt the agent, but speaking over the agent for >=192ms cuts off playback in <15ms.
- **Part A (Backchannel Test):**
  1. Ask Vaani: *"Can you tell me about the return policy?"*
  2. While Vaani is speaking, say a quick verbal nod: *"haan"* or *"hmm"*.
  3. Notice that Vaani **continues speaking without pausing or cutting off**, because the sound was under the 6-frame (192ms) backchannel threshold.
- **Part B (True Barge-In Test):**
  1. While Vaani is speaking mid-sentence, speak firmly:
     > *"Wait, stop! Cancel that and check my tickets instead."*
  2. Notice the **screen flashes red**, audio playback stops immediately (<15ms), a red event pill `⚡ Barge-In Detected (Interrupted Agent in <15ms)` appears in the log, and Vaani immediately begins processing your new request!

---

## 5. Troubleshooting & FAQ

- **Microphone permission denied:** Check the browser address bar icon and ensure `localhost:8000` has microphone permissions enabled. Use Chrome, Edge, or Brave.
- **Groq API 429 Rate Limit:** The orchestrator automatically catches 429 errors from `llama-3.3-70b-versatile` and falls back dynamically to `llama-3.1-8b-instant`.
- **Resetting Demo Data:** Click the **🔄 Reset State** button in the top right header at any time to restore the default sample appointments, tickets, and orders.
