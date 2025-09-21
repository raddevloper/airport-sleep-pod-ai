import os
import io
import cv2
import time
import math
import json
import sqlite3
import random
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

# Optional YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

# Optional MediaPipe
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False

# -----------------------------
# Paths & Globals
# -----------------------------
BASE = Path(__file__).parent
DATA = BASE / "data"
UPLOADS = DATA / "uploads"
PROCESSED = DATA / "processed"
DB_PATH = DATA / "smartpod.db"
FLIGHTS_CSV = DATA / "flights.csv"
TZ = pytz.timezone("Asia/Dubai")

for p in [DATA, UPLOADS, PROCESSED]:
    p.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Session-state + rerun helper
# -----------------------------
def ss_init(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

def safe_rerun():
    fn = getattr(st, "rerun", None)
    if callable(fn):
        fn()
        return
    fn2 = getattr(st, "experimental_rerun", None)
    if callable(fn2):
        fn2()

# -----------------------------
# Time utils
# -----------------------------
def now_tz():
    return dt.datetime.now(TZ)

def to_local(dt_naive_str: str):
    return TZ.localize(dt.datetime.strptime(dt_naive_str, "%Y-%m-%d %H:%M"))

def fmt_ts(ts):
    return ts.strftime("%Y-%m-%d %H:%M")

# -----------------------------
# Flights (fallback if no CSV)
# -----------------------------
def ensure_flights_csv():
    if FLIGHTS_CSV.exists():
        return
    today = now_tz().date()
    base_time = dt.datetime(today.year, today.month, today.day, 8, 0)
    airlines = [
        ("EK202", "Emirates", "DXB", "JFK", "B07"),
        ("QR101", "Qatar Airways", "DXB", "DOH", "C03"),
        ("EY18",  "Etihad", "DXB", "AUH", "D01"),
        ("TK761", "Turkish Airlines", "DXB", "IST", "A12"),
        ("BA106", "British Airways", "DXB", "LHR", "G02"),
        ("LH631", "Lufthansa", "DXB", "FRA", "H04"),
        ("AF655", "Air France", "DXB", "CDG", "J11"),
        ("PC741", "Pegasus", "DXB", "SAW", "K08"),
        ("SU521", "Aeroflot", "DXB", "SVO", "E09"),
        ("SV551", "Saudia", "DXB", "JED", "F05"),
    ]
    rows = []
    for i, (fn, al, orig, dest, gate) in enumerate(airlines):
        dep = base_time + dt.timedelta(minutes=60 * i)
        arr = dep + dt.timedelta(hours=random.randint(1, 8))
        status = random.choice(["On Time", "Boarding", "Delayed", "Gate Open"])
        rows.append({
            "flight_no": fn,
            "airline": al,
            "from": orig,
            "to": dest,
            "gate": gate,
            "sched_dep": dep.strftime("%Y-%m-%d %H:%M"),
            "sched_arr": arr.strftime("%Y-%m-%d %H:%M"),
            "status": status,
        })
    pd.DataFrame(rows).to_csv(FLIGHTS_CSV, index=False)

def load_flights():
    ensure_flights_csv()
    df = pd.read_csv(FLIGHTS_CSV)
    for c in ["flight_no", "airline", "from", "to", "gate", "status"]:
        df[c] = df[c].astype(str)
    return df

# -----------------------------
# Database (SQLite)
# -----------------------------
def db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    with db_conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name TEXT,
                last_name TEXT,
                email TEXT,
                selfie_path TEXT
            );
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS bookings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                flight_no TEXT,
                gate TEXT,
                created_at TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS status_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                ts TEXT,
                status TEXT,
                info TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
        """)

def add_user(first, last, email, selfie_path):
    with db_conn() as con:
        cur = con.execute(
            "INSERT INTO users(first_name, last_name, email, selfie_path) VALUES(?,?,?,?)",
            (first, last, email, str(selfie_path)),
        )
        return cur.lastrowid

def add_booking(user_id, flight_no, gate):
    with db_conn() as con:
        cur = con.execute(
            "INSERT INTO bookings(user_id, flight_no, gate, created_at) VALUES(?,?,?,?)",
            (user_id, flight_no, gate, fmt_ts(now_tz())),
        )
        return cur.lastrowid

def log_status(user_id, status, info=""):
    with db_conn() as con:
        con.execute(
            "INSERT INTO status_logs(user_id, ts, status, info) VALUES(?,?,?,?)",
            (user_id, fmt_ts(now_tz()), status, info),
        )

def get_users():
    with db_conn() as con:
        cur = con.execute("SELECT id, first_name, last_name, email, selfie_path FROM users")
        rows = cur.fetchall()
    cols = ["id", "first_name", "last_name", "email", "selfie_path"]
    return [dict(zip(cols, r)) for r in rows]

def get_user(user_id):
    with db_conn() as con:
        cur = con.execute("SELECT id, first_name, last_name, email, selfie_path FROM users WHERE id=?", (user_id,))
        row = cur.fetchone()
    if not row:
        return None
    cols = ["id", "first_name", "last_name", "email", "selfie_path"]
    return dict(zip(cols, row))

def get_bookings():
    with db_conn() as con:
        cur = con.execute("SELECT id, user_id, flight_no, gate, created_at FROM bookings")
        rows = cur.fetchall()
    cols = ["id", "user_id", "flight_no", "gate", "created_at"]
    return [dict(zip(cols, r)) for r in rows]

def get_last_status(user_id):
    with db_conn() as con:
        cur = con.execute(
            "SELECT ts, status, info FROM status_logs WHERE user_id=? ORDER BY id DESC LIMIT 1",
            (user_id,)
        )
        row = cur.fetchone()
    if not row:
        return None
    return {"ts": row[0], "status": row[1], "info": row[2]}

# -----------------------------
# Flight helpers
# -----------------------------
def minutes_to_departure_for_user(user_id, flights_df):
    bookings = [b for b in get_bookings() if b["user_id"] == user_id]
    if not bookings:
        return None
    b = bookings[-1]
    f = flights_df[flights_df["flight_no"] == b["flight_no"]]
    if f.empty:
        return None
    dep = to_local(str(f.iloc[0]["sched_dep"]))
    return (dep - now_tz()).total_seconds() / 60.0

def within_t30(minutes):
    return minutes is not None and 0.0 <= minutes <= 30.0

# -----------------------------
# BROWSER AUDIO + VOICE (Web Audio + Web Speech)
# -----------------------------
def sound_controls():
    """
    One-time Enable Sound button. Creates/resumes AudioContext and enables speechSynthesis.
    Exposes:
      - window.__smartpod_beep(freq, ms, vol)
      - window.__smartpod_say(text, rate, pitch)
    """
    components.html("""
    <div style="display:flex;gap:10px;align-items:center;margin:.4rem 0 .8rem;">
      <button id="sp-enable"
        style="padding:6px 10px;border-radius:8px;border:1px solid #555;background:#1f6feb;color:white;cursor:pointer">
        🔊 Enable Sound
      </button>
      <span id="sp-state" style="color:#9cdcfe;font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif">
        Sound: <strong>blocked</strong>
      </span>
    </div>
    <script>
    (function(){
      const KEY = "smartpod_sound_enabled";
      const stateEl = document.getElementById("sp-state");
      const btn = document.getElementById("sp-enable");
      let ctx = null;

      function status(){
        const ok = localStorage.getItem(KEY) === "1";
        stateEl.innerHTML = "Sound: <strong>" + (ok ? "enabled" : "blocked") + "</strong>";
      }
      function ensureCtx(){
        if (!ctx) { try { ctx = new (window.AudioContext || window.webkitAudioContext)(); } catch(_){} }
        return ctx;
      }
      window.__smartpod_beep = function(freq=1000, ms=600, vol=0.6){
        if (localStorage.getItem(KEY) !== "1") return false;
        const a = ensureCtx(); if (!a) return false;
        const osc = a.createOscillator();
        const gain = a.createGain();
        osc.type="sine"; osc.frequency.setValueAtTime(freq, a.currentTime);
        gain.gain.setValueAtTime(vol, a.currentTime);
        osc.connect(gain).connect(a.destination);
        osc.start(); osc.stop(a.currentTime + (ms/1000));
        return true;
      };
      window.__smartpod_say = function(text="Wake up!", rate=1.0, pitch=1.0){
        if (localStorage.getItem(KEY) !== "1") return false;
        try{
          const u = new SpeechSynthesisUtterance(text);
          u.rate = rate; u.pitch = pitch;
          // pick a voice if available (prefer English)
          const voices = window.speechSynthesis.getVoices();
          const en = voices.find(v => /en/i.test(v.lang));
          if (en) u.voice = en;
          window.speechSynthesis.cancel(); // avoid queue pile-up
          window.speechSynthesis.speak(u);
          return true;
        }catch(e){ return false; }
      };
      btn.addEventListener("click", async ()=>{
        const a = ensureCtx();
        try{
          if (a && a.state === "suspended") await a.resume();
          localStorage.setItem(KEY, "1");
          status();
          // Confirmation blip + short “ready”
          window.__smartpod_beep(880, 180, 0.55);
          setTimeout(()=>{ try{ window.__smartpod_say("Sound enabled"); }catch(e){} }, 120);
        }catch(e){ console.warn("Audio resume failed", e); }
      });
      // Some browsers load voices asynchronously
      if (window.speechSynthesis && window.speechSynthesis.onvoiceschanged === null) {
        window.speechSynthesis.onvoiceschanged = function(){};
      }
      status();
    })();
    </script>
    """, height=60)

def browser_beep(freq=1000, duration_ms=700, volume=0.65):
    st.markdown(f"""
    <script>
      (function(){{
        if (window.__smartpod_beep) {{
          const ok = window.__smartpod_beep({freq}, {duration_ms}, {volume});
          if (!ok) {{
            const el = document.getElementById('sp-audio-hint');
            if (el) el.style.display = 'block';
          }}
        }}
      }})();
    </script>
    """, unsafe_allow_html=True)

def browser_say(text, rate=1.0, pitch=1.0):
    # trigger speech; if blocked, show hint
    js = f"""
    <script>
      (function(){{
        if (window.__smartpod_say) {{
          const ok = window.__smartpod_say({json.dumps(text)}, {rate}, {pitch});
          if (!ok) {{
            const el = document.getElementById('sp-audio-hint');
            if (el) el.style.display = 'block';
          }}
        }}
      }})();
    </script>
    """
    st.markdown(js, unsafe_allow_html=True)
    st.markdown("""
    <div id="sp-audio-hint" style="display:none;color:#f0ad4e;margin:.5rem 0;">
      🔇 Audio blocked. Click <strong>“🔊 Enable Sound”</strong> above once, then try again.
    </div>
    """, unsafe_allow_html=True)

def wake_sound():
    browser_beep(freq=1200, duration_ms=800, volume=0.75)
    browser_say("Wake up! Your flight is near. Please wake up now.", rate=1.0, pitch=1.0)

def success_sound():
    browser_beep(freq=700, duration_ms=300, volume=0.55)
    browser_say("Good luck! You're ready to go.", rate=1.05, pitch=1.0)

# -----------------------------
# Vision: Eye Aspect Ratio
# -----------------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def ear_val_from_pts(pts):
    return (dist(pts[1], pts[5]) + dist(pts[2], pts[4])) / (2.0 * dist(pts[0], pts[3]) + 1e-6)

def process_video(video_path, selfie_path=None, use_yolo=False, max_seconds=30):
    if not MP_AVAILABLE:
        raise RuntimeError("MediaPipe not available. Please `pip install mediapipe`.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT), max_seconds * fps))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = PROCESSED / f"processed_{int(time.time())}.mp4"
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    yolo_model = None
    if use_yolo and YOLO_AVAILABLE:
        try:
            yolo_model = YOLO("yolov8n.pt")
        except Exception:
            yolo_model = None

    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    consec_closed = 0
    max_consec_closed = 0
    statuses = []

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        status = "AWAKE"
        ear_avg = None

        if res.multi_face_landmarks:
            h_, w_, _ = frame.shape
            lm = res.multi_face_landmarks[0].landmark

            def pts(idx_list):
                return [(int(lm[j].x * w_), int(lm[j].y * h_)) for j in idx_list]

            left_pts = pts(LEFT_EYE)
            right_pts = pts(RIGHT_EYE)
            ear_l = ear_val_from_pts(left_pts)
            ear_r = ear_val_from_pts(right_pts)
            ear_avg = (ear_l + ear_r) / 2.0

            if ear_avg < 0.24:
                status = "SLEEPY"
                consec_closed += 1
                max_consec_closed = max(max_consec_closed, consec_closed)
            else:
                consec_closed = 0

            for (x, y) in left_pts + right_pts:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        statuses.append(status)

        cv2.putText(frame, f"Status: {status}" + (f" | EAR={ear_avg:.2f}" if ear_avg else ""),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.putText(frame, f"Status: {status}" + (f" | EAR={ear_avg:.2f}" if ear_avg else ""),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        writer.write(frame)

    cap.release()
    writer.release()

    duration = total_frames / (fps or 25)
    sleepy_ratio = statuses.count("SLEEPY") / max(len(statuses), 1)
    max_consec_sec = max_consec_closed / (fps or 25)

    summary = {
        "fps": float(fps),
        "frames": int(total_frames),
        "duration_sec": float(duration),
        "sleepy_ratio": float(sleepy_ratio),
        "max_consec_sleep_sec": float(max_consec_sec),
        "final_status": "SLEEPY" if sleepy_ratio > 0.4 or max_consec_sec > 3 else "AWAKE",
    }
    return out_path, summary

# -----------------------------
# Alerts logic
# -----------------------------
def compute_alerts(flights_df):
    alerts = []
    bookings = get_bookings()
    users = {u["id"]: u for u in get_users()}
    for b in bookings:
        u = users.get(b["user_id"])
        frow = flights_df[flights_df["flight_no"] == b["flight_no"]]
        if frow.empty:
            continue
        dep_str = frow.iloc[0]["sched_dep"]
        dep_ts = to_local(str(dep_str)) if isinstance(dep_str, str) else dep_str
        minutes = (dep_ts - now_tz()).total_seconds() / 60.0
        last = get_last_status(b["user_id"]) or {"status": "AWAKE", "ts": None}
        if 0 <= minutes <= 30 and last["status"] == "SLEEPY":
            alerts.append({
                "user": f"{u['first_name']} {u['last_name']}" if u else f"User {b['user_id']}",
                "flight_no": b["flight_no"],
                "gate": b["gate"],
                "dep_ts": dep_ts,
                "minutes_to_dep": round(minutes, 1),
                "last_status": last["status"],
                "last_seen": last["ts"],
            })
    return alerts

# -----------------------------
# UI
# -----------------------------
def ui_board(flights_df):
    st.subheader("🛫 Flight Board (Today – Asia/Dubai)")
    flights_df = flights_df.copy()
    flights_df["sched_dep_dt"] = flights_df["sched_dep"].apply(lambda s: to_local(str(s)))
    flights_df = flights_df.sort_values("sched_dep_dt")
    show = flights_df[["flight_no", "airline", "from", "to", "gate", "sched_dep", "status"]]
    st.dataframe(show, use_container_width=True, hide_index=True)

def ui_checkin(flights_df):
    st.subheader("🛌 Pod Check-In (Traveler)")

    ss_init("checkin_first", "")
    ss_init("checkin_last", "")
    ss_init("checkin_email", "")
    first_flight = flights_df["flight_no"].iloc[0] if not flights_df.empty else ""
    ss_init("checkin_flight", first_flight)

    def gate_for(fno):
        try:
            return str(flights_df.loc[flights_df.flight_no == fno, "gate"].iloc[0])
        except Exception:
            return ""

    if "checkin_gate" not in st.session_state:
        st.session_state["checkin_gate"] = gate_for(st.session_state["checkin_flight"])

    with st.form("checkin_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        st.session_state["checkin_first"] = c1.text_input("First name", value=st.session_state["checkin_first"])
        st.session_state["checkin_last"]  = c2.text_input("Last name",  value=st.session_state["checkin_last"])
        st.session_state["checkin_email"] = st.text_input("Email (optional)", value=st.session_state["checkin_email"])

        flights_list = flights_df["flight_no"].tolist()
        try:
            idx = flights_list.index(st.session_state["checkin_flight"])
        except ValueError:
            idx = 0
        new_flight = st.selectbox("Your flight", flights_list, index=idx, key="checkin_flight")
        if new_flight != st.session_state.get("_last_flight"):
            st.session_state["checkin_gate"] = gate_for(new_flight)
            st.session_state["_last_flight"] = new_flight

        st.session_state["checkin_gate"] = st.text_input("Gate", value=st.session_state["checkin_gate"])
        selfie = st.file_uploader("Upload selfie (jpg/png)", type=["jpg", "jpeg", "png"], key="checkin_selfie")

        submitted = st.form_submit_button("Save Check-In")
        if submitted:
            if not st.session_state["checkin_first"] or not st.session_state["checkin_last"] or not selfie:
                st.error("Please provide first name, last name, and a selfie.")
            else:
                img = Image.open(selfie).convert("RGB")
                fname = f"selfie_{int(time.time())}.jpg"
                path = UPLOADS / fname
                img.save(path)

                uid = add_user(
                    st.session_state["checkin_first"],
                    st.session_state["checkin_last"],
                    st.session_state["checkin_email"],
                    path
                )
                bid = add_booking(uid, st.session_state["checkin_flight"], st.session_state["checkin_gate"])
                st.success(f"Saved! User #{uid} — Booking #{bid} for flight {st.session_state['checkin_flight']} at gate {st.session_state['checkin_gate']}.")
                st.image(str(path), caption="Selfie saved", width=220)

def ui_monitor():
    st.subheader("🎥 Pod Monitoring (Upload short clip – up to ~30s)")
    sound_controls()

    users = get_users()
    if not users:
        st.info("No users yet. Please check-in first.")
        return

    user_map = {f"{u['first_name']} {u['last_name']} (#{u['id']})": u["id"] for u in users}
    sel = st.selectbox("Select traveler", list(user_map.keys()), key="monitor_user_sel")
    uid = user_map[sel]

    use_yolo = st.checkbox("Use YOLO person overlay (requires ultralytics)", value=False, key="monitor_yolo")

    ss_init("monitor_video_bytes", None)
    uploaded = st.file_uploader("Upload MP4/MOV", type=["mp4", "mov", "m4v"], key="monitor_uploader")
    if uploaded is not None:
        st.session_state["monitor_video_bytes"] = uploaded.read()

    if st.session_state["monitor_video_bytes"] is not None:
        tmp_path = UPLOADS / f"video_{uid}_{int(time.time())}.mp4"
        with open(tmp_path, "wb") as f:
            f.write(st.session_state["monitor_video_bytes"])
        st.video(str(tmp_path))

        if st.button("Analyze video", key="monitor_analyze"):
            with st.spinner("Analyzing..."):
                try:
                    out_path, summary = process_video(tmp_path, None, use_yolo=use_yolo)
                except Exception as e:
                    st.error(f"Vision error: {e}")
                    return
            st.success("Done!")
            st.video(str(out_path))
            cols = st.columns(3)
            cols[0].metric("Final Status", summary["final_status"])
            cols[1].metric("Max Sleep Streak (s)", f"{summary['max_consec_sleep_sec']:.1f}")
            cols[2].metric("Sleepy %", f"{summary['sleepy_ratio']*100:.1f}%")

            flights_df = load_flights()
            mins = minutes_to_departure_for_user(uid, flights_df)
            if summary["final_status"] == "SLEEPY" and within_t30(mins):
                st.error(f"🚨 WAKE UP! Flight in {mins:.1f} min.")
                wake_sound()
            else:
                if summary["final_status"] == "SLEEPY":
                    st.warning("SLEEPY detected, but not inside T-30 window — no voice alert.")
                else:
                    st.info("AWAKE — no alert.")

            log_status(uid, summary["final_status"], info=json.dumps(summary))
    else:
        st.info("Upload a short clip to begin.")

def ui_alerts(_):
    st.subheader("⏰ Alerts (≤ 30 min to departure & traveler is SLEEPY)")
    col1, _ = st.columns([1,2])
    if col1.button("Refresh alerts"):
        safe_rerun()

    flights_df = load_flights()
    st.write(f"**Now (Asia/Dubai):** {fmt_ts(now_tz())}")

    alerts = compute_alerts(flights_df)
    if alerts:
        for a in alerts:
            st.error(
                f"**Wake up {a['user']}!** Flight **{a['flight_no']}** (Gate **{a['gate']}**) "
                f"departs in **{a['minutes_to_dep']} min**. Last status: **{a['last_status']}** at {a['last_seen']}."
            )
    else:
        st.success("No alerts right now.")

# -----------------------------
# Live Webcam Monitor
# -----------------------------
def ui_live_monitor():
    st.subheader("🟢 Live Webcam Monitor (EAR-based sleep detection)")
    sound_controls()

    if not MP_AVAILABLE:
        st.error("MediaPipe not available. Please `pip install mediapipe`.")
        return

    users = get_users()
    if not users:
        st.info("No users yet. Please check-in first.")
        return

    user_map = {f"{u['first_name']} {u['last_name']} (#{u['id']})": u["id"] for u in users}
    sel = st.selectbox("Traveler", list(user_map.keys()), key="live_user_sel")
    uid = user_map[sel]

    c1, c2, c3 = st.columns(3)
    run_btn = c1.button("Start")
    stop_btn = c2.button("Stop")
    respect_t30 = c3.toggle("Respect T-30 window", value=True,
                            help="Only alert if the traveler’s flight departs in 0–30 minutes. Turn off to test.")

    ear_thresh = st.slider("Eye closure threshold (EAR = Eye Aspect Ratio)", 0.15, 0.35, 0.24, 0.01)
    min_streak = st.slider("Trigger if eyes closed for (seconds)", 1.0, 5.0, 2.0, 0.5)

    frame_slot = st.empty()
    status_slot = st.empty()
    window_slot = st.empty()
    alert_slot = st.empty()

    ss_init("live_running", False)
    ss_init("alert_active", False)
    ss_init("awake_streak_frames", 0)

    if run_btn:
        st.session_state.live_running = True
        st.session_state.alert_active = False
        st.session_state.awake_streak_frames = 0
        alert_slot.empty()

    if stop_btn:
        st.session_state.live_running = False
        alert_slot.info("Stopped.")

    if not st.session_state.live_running:
        st.info("Press **Start** to access your webcam.")
        return

    flights_df = load_flights()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam. Grant camera permission to your terminal/IDE.")
        st.session_state.live_running = False
        return

    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    def ear_from_landmarks(lm, w, h, idxs):
        pts = [(int(lm[j].x * w), int(lm[j].y * h)) for j in idxs]
        def d(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])
        return (d(pts[1], pts[5]) + d(pts[2], pts[4])) / (2.0 * d(pts[0], pts[3]) + 1e-6), pts

    consec_closed = 0
    fps_est = 20.0
    t_prev = time.time()

    try:
        while st.session_state.live_running:
            ret, frame = cap.read()
            if not ret:
                break

            t_now = time.time()
            dt_s = max(1e-3, t_now - t_prev)
            fps_est = 0.9 * fps_est + 0.1 * (1.0 / dt_s)
            t_prev = t_now

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            status = "AWAKE"
            ear_val = None

            if res.multi_face_landmarks:
                h, w, _ = frame.shape
                lm = res.multi_face_landmarks[0].landmark

                ear_l, lpts = ear_from_landmarks(lm, w, h, LEFT_EYE)
                ear_r, rpts = ear_from_landmarks(lm, w, h, RIGHT_EYE)
                ear_val = (ear_l + ear_r) / 2.0

                for (x, y) in lpts + rpts:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                if ear_val < ear_thresh:
                    status = "SLEEPY"
                    consec_closed += 1
                else:
                    consec_closed = 0

            cv2.putText(frame, f"Status: {status}" + (f" | Eye Aspect Ratio (EAR)={ear_val:.2f}" if ear_val else ""),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
            cv2.putText(frame, f"Status: {status}" + (f" | Eye Aspect Ratio (EAR)={ear_val:.2f}" if ear_val else ""),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            frame_slot.image(frame, channels="BGR")

            mins = minutes_to_departure_for_user(uid, flights_df)
            if mins is None:
                window_slot.warning("No booking / flight time found for this traveler.")
            else:
                window_slot.info(f"Minutes to departure: **{mins:.1f}**  |  T-30 window: **{within_t30(mins)}**")

            need_frames = int(min_streak * max(8.0, fps_est))
            status_slot.write(
                f"Closed-eye frames: **{consec_closed}** / {need_frames}  |  FPS≈{fps_est:.1f}"
            )

            should_alert = (consec_closed >= need_frames)
            if respect_t30:
                should_alert = should_alert and within_t30(mins)

            if should_alert and not st.session_state.alert_active:
                msg = "🚨 WAKE UP!"
                if mins is not None:
                    msg += f" Flight in {mins:.1f} min."
                alert_slot.error(msg)
                wake_sound()
                st.session_state.alert_active = True
                st.session_state.awake_streak_frames = 0

            if st.session_state.alert_active:
                if status == "AWAKE":
                    st.session_state.awake_streak_frames += 1
                    if st.session_state.awake_streak_frames >= int(max(8.0, fps_est) * 1.0):
                        alert_slot.success("✅ Good luck!")
                        success_sound()
                        st.session_state.alert_active = False
                        st.session_state.awake_streak_frames = 0
                else:
                    st.session_state.awake_streak_frames = 0

            time.sleep(0.01)

    finally:
        cap.release()

# -----------------------------
# Main app
# -----------------------------
def main():
    st.set_page_config(page_title="SmartPod Demo", page_icon="😴", layout="wide")
    st.title("😴✈️ SmartPod Demo – Sleep Pod Monitoring + Flight Alerts")
    st.caption("Flight board, traveler check-in with selfie, uploaded video analysis, LIVE webcam with T-30 logic + browser audio & voice alerts, and T-30 departure alerts.")

    init_db()
    flights_df = load_flights()

    tabs = st.tabs(["Flight Board", "Pod Check-In", "Pod Monitor", "Alerts", "Live Monitor"])
    with tabs[0]:
        ui_board(flights_df)
    with tabs[1]:
        ui_checkin(flights_df)
    with tabs[2]:
        ui_monitor()
    with tabs[3]:
        ui_alerts(flights_df)
    with tabs[4]:
        ui_live_monitor()

    st.sidebar.markdown("### Tips")
    st.sidebar.write("• Click **🔊 Enable Sound** once per tab to unlock audio/voice.")
    st.sidebar.write("• Turn OFF **Respect T-30 window** to test alerts immediately.")
    st.sidebar.write("• Alert auto-clears when you’re awake ~1s (says “Good luck!”).")

if __name__ == "__main__":
    main()
