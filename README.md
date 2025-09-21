# 😴✈️ SmartPod Demo — Sleep Pod Monitoring + Flight Alerts

A Streamlit demo that simulates an airport “sleep pod” system:

- 📋 **Flight board** (CSV-driven)
- 🧍 **Traveler check-in** with selfie (saved locally)
- 🎥 **Pod monitor**: upload a short video; detect **AWAKE/SLEEPY**
- 📸 **Live webcam monitor** using **EAR (Eye Aspect Ratio)** with thresholds
- ⏰ **T-30 alerts**: if flight departs within 30 minutes **and** traveler is sleepy
- 🔊 In-browser **audio + voice alerts** (WebAudio + Web Speech API)  
  → “Wake up!” when sleepy, “Good luck!” when awake again

> No cloud dependencies. Everything runs locally.

---

## 📦 Features

- **Streamlit UI** with tabs: Flight Board, Pod Check-In, Pod Monitor, Alerts, Live Monitor
- **EAR-based eye closure** from **MediaPipe Face Mesh** (no model training)
- Optional **YOLOv8** person overlay on uploaded videos (requires `ultralytics`)
- **Audio in browser** (works across Chrome/Firefox/Safari after a one-time “Enable Sound” click)
- **SQLite** for check-ins + status logs
- **Asia/Dubai timezone** used for flight times (configurable in code)
🧭 How to use

Flight Board: shows flights from data/flights.csv.
You can edit this file to add or change flights/times.

Pod Check-In: enter first/last name, email (optional), choose flight & gate, upload a selfie → Save.

Pod Monitor (upload): pick a traveler, upload a short MP4/MOV, click Analyze.
If final status is SLEEPY and within 30 minutes of departure, you’ll hear “Wake up!”

Live Monitor (webcam):

Click 🔊 Enable Sound once (you should hear “Sound enabled”).

(For testing) toggle Respect T-30 window OFF to trigger instantly.

Click Start, then close your eyes for the configured seconds (slider).

You’ll see 🚨 WAKE UP! + hear voice. When you open your eyes briefly, it clears with “Good luck!”.

🔊 Browser audio tips

Chrome/Firefox/Safari: After clicking 🔊 Enable Sound, audio + voice should work.

If you still don’t hear voice:

In Safari: Settings → Websites → Auto-Play → localhost → Allow All Auto-Play

Use the 🔈 Test Voice button (shown near “Enable Sound”) to confirm voice is available.

🧪 EAR (Eye Aspect Ratio) notes

EAR is a geometry metric measuring eye openness.
Typical open values: ~0.28–0.35; closed: < ~0.22.

Tune the threshold and closed-eye seconds in Live Monitor to your face/camera.

⚙️ Configuration

Timezone (default Asia/Dubai) is set in app_streamlit.py (TZ = pytz.timezone("Asia/Dubai")).

Data folders: data/uploads/ and data/processed/ are created automatically.

Database: data/smartpod.db (SQLite) stores users, bookings, and status logs.# airport-sleep-pod-ai
