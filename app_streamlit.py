import os
import io
import cv2
import json
import time
import math
import sqlite3
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

# Cloud-safe live webcam (browser camera via WebRTC)
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av

# ----- Optional YOLO (disabled on Cloud) -----
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

# ----- MediaPipe Face Mesh -----
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False

# =============================================================================
# Paths / Globals
# =============================================================================
BASE = Path(__file__).parent
DATA = BASE / "data"
UPLOADS = DATA / "uploads"
PROCESSED = DATA / "processed"
DB_PATH = DATA / "smartpod.db"
FLIGHTS_CSV = DATA / "flights.csv"
TZ = pytz.timezone("Asia/Dubai")

for p in (DATA, UPLOADS, PROCESSED):
    p.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Session helpers
# =============================================================================
def sget(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

def safe_rerun():
    fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if callable(fn):
        fn()

# =============================================================================
# Time helpers
# =============================================================================
def now_tz():
    return dt.datetime.now(TZ)

def fmt_ts(ts: dt.datetime) -> str:
    return ts.strftime("%Y-%m-%d %H:%M")

def to_local(s: str) -> dt.datetime:
    # s format: "YYYY-MM-DD HH:MM"
    return TZ.localize(dt.datetime.strptime(s, "%Y-%m-%d %H:%M"))

# =============================================================================
# Flights (CSV)
# =============================================================================
def ensure_flights_csv():
    if FLIGHTS_CSV.exists():
        return
    # create a simple schedule (1 per hour)
    today = now_tz().date()
    base = dt.datetime(today.year, today.month, today.day, 8, 0)
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
    for i, (fn, al, fr, to, gate) in enumerate(airlines):
        dep = base + dt.timedelta(minutes=60 * i)
        arr = dep + dt.timedelta(hours=4)
        rows.append({
            "flight_no": fn, "airline": al, "from": fr, "to": to, "gate": gate,
            "sched_dep": dep.strftime("%Y-%m-%d %H:%M"),
            "sched_arr": arr.strftime("%Y-%m-%d %H:%M"),
            "status": "On Time"
        })
    pd.DataFrame(rows).to_csv(FLIGHTS_CSV, index=False)

def load_flights() -> pd.DataFrame:
    ensure_flights_csv()
    df = pd.read_csv(FLIGHTS_CSV)
    for c in ["flight_no", "airline", "from", "to", "gate", "status"]:
        df[c] = df[c].astype(str)
    return df

# =============================================================================
# SQLite (users / bookings / status logs)
# =============================================================================
def db_conn():
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA foreign_keys = ON;")
    return con

def init_db():
    with db_conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS users(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              first_name TEXT, last_name TEXT, email TEXT, selfie_path TEXT
            );
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS bookings(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER, flight_no TEXT, gate TEXT, created_at TEXT,
              FOREIGN KEY(user_id) REFERENCES users(id)
            );
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS status_logs(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER, ts TEXT, status TEXT, info TEXT,
              FOREIGN KEY(user_id) REFERENCES users(id)
            );
        """)

def add_user(first, last, email, selfie_path):
    with db_conn() as con:
        cur = con.execute(
            "INSERT INTO users(first_name,last_name,email,selfie_path) VALUES(?,?,?,?)",
            (first, last, email, str(selfie_path))
        )
        return cur.lastrowid

def add_booking(user_id, flight_no, gate):
    with db_conn() as con:
        cur = con.execute(
            "INSERT INTO bookings(user_id,flight_no,gate,created_at) VALUES(?,?,?,?)",
            (user_id, flight_no, gate, fmt_ts(now_tz()))
        )
        return cur.lastrowid

def log_status(user_id, status, info=""):
    with db_conn() as con:
        con.execute(
            "INSERT INTO status_logs(user_id,ts,status,info) VALUES(?,?,?,?)",
            (user_id, fmt_ts(now_tz()), status, info)
        )

def get_users():
    with db_conn() as con:
        cur = con.execute("SELECT id,first_name,last_name,email,selfie_path FROM users")
        rows = cur.fetchall()
    cols = ["id","first_name","last_name","email","selfie_path"]
    return [dict(zip(cols,r)) for r in rows]

def get_last_status(user_id):
    with db_conn() as con:
        cur = con.execute(
            "SELECT ts,status,info FROM status_logs WHERE user_id=? ORDER BY id DESC LIMIT 1",
            (user_id,)
        )
        row = cur.fetchone()
    if not row:
        return None
    return {"ts": row[0], "status": row[1], "info": row[2]}

def get_bookings():
    with db_conn() as con:
        cur = con.execute("SELECT id,user_id,flight_no,gate,created_at FROM bookings")
        rows = cur.fetchall()
    cols = ["id","user_id","flight_no","gate","created_at"]
    return [dict(zip(cols,r)) for r in rows]

# =============================================================================
# Flight helpers / alerts
# =============================================================================
def minutes_to_departure_for_user(user_id, flights_df: pd.DataFrame):
    # get last booking for user
    bookings = [b for b in get_bookings() if b["user_id"] == user_id]
    if not bookings:
        return None
    flight_no = bookings[-1]["flight_no"]
    f = flights_df[flights_df["flight_no"] == flight_no]
    if f.empty:
        return None
    dep = to_local(str(f.iloc[0]["sched_dep"]))
    return (dep - now_tz()).total_seconds() / 60.0

def within_t30(mins):
    return mins is not None and 0.0 <= mins <= 30.0

def compute_alerts(flights_df):
    alerts = []
    users = {u["id"]: u for u in get_users()}
    for b in get_bookings():
        u = users.get(b["user_id"])
        frow = flights_df[flights_df["flight_no"] == b["flight_no"]]
        if frow.empty:
            continue
        dep = to_local(str(frow.iloc[0]["sched_dep"]))
        mins = (dep - now_tz()).total_seconds() / 60.0
        last = get_last_status(b["user_id"]) or {"status": "AWAKE", "ts": None}
        if within_t30(mins) and last["status"] == "SLEEPY":
            alerts.append({
                "user": f"{u['first_name']} {u['last_name']}" if u else f"User {b['user_id']}",
                "flight_no": b["flight_no"],
                "gate": b["gate"],
                "minutes": round(mins,1),
                "last": last
            })
    return alerts

# =============================================================================
# Browser Audio + Voice (WebAudio + Web Speech API)
# =============================================================================
def sound_controls():
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
      const KEY="smartpod_sound_enabled";
      const stateEl=document.getElementById("sp-state");
      let ctx=null;
      function status(){
        const ok=localStorage.getItem(KEY)==="1";
        stateEl.innerHTML="Sound: <strong>"+(ok?"enabled":"blocked")+"</strong>";
      }
      function ensureCtx(){
        if(!ctx){try{ctx=new (window.AudioContext||window.webkitAudioContext)();}catch(e){}}
        return ctx;
      }
      window.__smartpod_beep=function(freq=1000,ms=600,vol=0.6){
        if(localStorage.getItem(KEY)!=="1")return false;
        const a=ensureCtx(); if(!a) return false;
        const o=a.createOscillator(); const g=a.createGain();
        o.type="sine"; o.frequency.setValueAtTime(freq,a.currentTime);
        g.gain.setValueAtTime(vol,a.currentTime);
        o.connect(g).connect(a.destination); o.start(); o.stop(a.currentTime+(ms/1000));
        return true;
      };
      window.__smartpod_say=function(text="Wake up!",rate=1.0,pitch=1.0){
        if(localStorage.getItem(KEY)!=="1")return false;
        try{ const u=new SpeechSynthesisUtterance(text);
             u.rate=rate; u.pitch=pitch;
             const vs=window.speechSynthesis.getVoices()||[];
             const en=vs.find(v=>/en/i.test(v.lang)); if(en) u.voice=en;
             window.speechSynthesis.cancel(); window.speechSynthesis.speak(u); return true;
        }catch(e){ return false; }
      };
      document.getElementById("sp-enable").addEventListener("click", async ()=>{
        const a=ensureCtx(); try{ if(a && a.state==="suspended") await a.resume(); }catch(e){}
        localStorage.setItem(KEY,"1"); status();
        if(window.__smartpod_beep) window.__smartpod_beep(880,180,0.55);
        setTimeout(()=>{ try{ window.__smartpod_say("Sound enabled"); }catch(e){} }, 120);
      });
      if(window.speechSynthesis && window.speechSynthesis.onvoiceschanged===null){
        window.speechSynthesis.onvoiceschanged=function(){};
      }
      status();
    })();
    </script>
    """, height=60)

def voice_test_widget():
    components.html("""
    <div style="display:flex;gap:8px;align-items:center;margin:.25rem 0;">
      <button id="sp-voice-test"
        style="padding:6px 10px;border-radius:8px;border:1px solid #555;background:#0ea5e9;color:white;cursor:pointer">
        🔈 Test Voice
      </button>
      <span id="sp-voice-info" style="color:#9cdcfe;font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif">
        Voices: <em>detecting…</em>
      </span>
    </div>
    <script>
    (function(){
      const info=document.getElementById('sp-voice-info');
      function list(){ try{ const v=window.speechSynthesis?window.speechSynthesis.getVoices():[]; info.innerHTML="Voices: <strong>"+(v?v.length:0)+"</strong>"; }catch(e){ info.textContent="Voices: not available"; } }
      list(); if(window.speechSynthesis && window.speechSynthesis.onvoiceschanged===null){ window.speechSynthesis.onvoiceschanged=list; }
      document.getElementById('sp-voice-test').addEventListener('click', ()=>{
        try{ const u=new SpeechSynthesisUtterance("This is a voice test."); const vs=window.speechSynthesis.getVoices()||[]; const en=vs.find(v=>/en/i.test(v.lang)); if(en) u.voice=en; window.speechSynthesis.cancel(); window.speechSynthesis.speak(u); }catch(e){ alert("Speech synthesis not available: "+e); }
      });
    })();
    </script>
    """, height=60)

def browser_beep(freq=1000, duration_ms=700, volume=0.65):
    st.markdown(f"""
    <script>
      (function(){{ if(window.__smartpod_beep) {{ window.__smartpod_beep({freq}, {duration_ms}, {volume}); }} }})();
    </script>
    """, unsafe_allow_html=True)

def browser_say(text, rate=1.0, pitch=1.0):
    js = f"""
    <script>
      (function(){{
        function speakOnce(){{
          try{{ const u=new SpeechSynthesisUtterance({json.dumps(text)});
                u.rate={rate}; u.pitch={pitch};
                const s=window.speechSynthesis; const vs=s.getVoices()||[];
                const en=vs.find(v=>/en/i.test(v.lang)); if(en) u.voice=en;
                s.cancel(); s.speak(u); return true; }}catch(e){{ return false; }}
        }}
        let tries=0; (function waitVoices(){{ tries++; 
           if((window.speechSynthesis && window.speechSynthesis.getVoices().length) || tries>10){{ speakOnce(); }}
           else setTimeout(waitVoices,200);
        }})();
      }})();
    </script>
    """
    st.markdown(js, unsafe_allow_html=True)

def wake_sound():
    browser_beep(1200, 800, 0.75)
    browser_say("Wake up! Your flight is near. Please wake up now.", 1.0, 1.0)

def success_sound():
    browser_beep(700, 300, 0.55)
    browser_say("Good luck! You're ready to go.", 1.05, 1.0)

# =============================================================================
# EAR (Eye Aspect Ratio)
# =============================================================================
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def _d(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def ear_from_pts(pts):
    return (_d(pts[1], pts[5]) + _d(pts[2], pts[4])) / (2.0 * _d(pts[0], pts[3]) + 1e-6)

# =============================================================================
# Video (upload) analysis
# =============================================================================
def process_video(video_path, use_yolo=False, max_seconds=30):
    if not MP_AVAILABLE:
        raise RuntimeError("MediaPipe not available on this deployment.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT), max_seconds * fps))

    out_path = PROCESSED / f"processed_{int(time.time())}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    yolo_model = None
    if use_yolo and YOLO_AVAILABLE:
        try:
            yolo_model = YOLO("yolov8n.pt")
        except Exception:
            yolo_model = None

    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    consec_closed = 0
    max_consec = 0
    tags = []

    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        status = "AWAKE"
        ear_val = None

        if res.multi_face_landmarks:
            H, W, _ = frame.shape
            lm = res.multi_face_landmarks[0].landmark
            def P(idxs): return [(int(lm[j].x*W), int(lm[j].y*H)) for j in idxs]
            L = P(LEFT_EYE); R = P(RIGHT_EYE)
            ear_val = (ear_from_pts(L) + ear_from_pts(R)) / 2.0
            if ear_val < 0.24:
                status = "SLEEPY"
                consec_closed += 1
                max_consec = max(max_consec, consec_closed)
            else:
                consec_closed = 0
            for x, y in L+R:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        if yolo_model is not None:
            try:
                for r in yolo_model.predict(frame, imgsz=640, conf=0.25, verbose=False):
                    for box in r.boxes.xyxy.cpu().numpy():
                        x1,y1,x2,y2 = map(int, box[:4])
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
            except Exception:
                pass

        cv2.putText(frame, f"Status: {status}" + (f" | EAR={ear_val:.2f}" if ear_val else ""),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3)
        cv2.putText(frame, f"Status: {status}" + (f" | EAR={ear_val:.2f}" if ear_val else ""),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        tags.append(status)
        writer.write(frame)

    cap.release(); writer.release()

    sleepy_ratio = tags.count("SLEEPY") / max(len(tags), 1)
    max_consec_sec = max_consec / (fps or 25)
    summary = {
        "fps": float(fps),
        "frames": int(total_frames),
        "duration_sec": float(total_frames / (fps or 25)),
        "sleepy_ratio": float(sleepy_ratio),
        "max_consec_sleep_sec": float(max_consec_sec),
        "final_status": "SLEEPY" if sleepy_ratio > 0.4 or max_consec_sec > 3 else "AWAKE",
    }
    return out_path, summary

# =============================================================================
# UI: Flight board / Check-in / Monitor / Alerts / Live (WebRTC)
# =============================================================================
def ui_board(flights_df):
    st.subheader("🛫 Flight Board (Today – Asia/Dubai)")
    df = flights_df.copy()
    df["sched_dep_dt"] = df["sched_dep"].apply(lambda s: to_local(str(s)))
    df = df.sort_values("sched_dep_dt")
    st.dataframe(df[["flight_no","airline","from","to","gate","sched_dep","status"]],
                 use_container_width=True, hide_index=True)

def ui_checkin(flights_df):
    st.subheader("🛌 Pod Check-In (Traveler)")

    sget("first",""); sget("last",""); sget("email","")
    flights = flights_df["flight_no"].tolist()
    pick = flights[0] if flights else ""
    sget("flight", pick)

    def gate_for(fno):
        try: return str(flights_df.loc[flights_df.flight_no==fno,"gate"].iloc[0])
        except Exception: return ""

    if "gate" not in st.session_state:
        st.session_state.gate = gate_for(sget("flight", pick))

    with st.form("checkin", clear_on_submit=False):
        c1, c2 = st.columns(2)
        st.session_state.first = c1.text_input("First name", st.session_state.first)
        st.session_state.last  = c2.text_input("Last name",  st.session_state.last)
        st.session_state.email = st.text_input("Email (optional)", st.session_state.email)

        try: idx = flights.index(st.session_state.flight)
        except ValueError: idx = 0
        st.session_state.flight = st.selectbox("Your flight", flights, index=idx)
        st.session_state.gate = st.text_input("Gate", value=st.session_state.gate or gate_for(st.session_state.flight))
        selfie = st.file_uploader("Upload selfie (jpg/png)", type=["jpg","jpeg","png"])

        if st.form_submit_button("Save Check-In"):
            if not st.session_state.first or not st.session_state.last or not selfie:
                st.error("Please provide first name, last name, and a selfie.")
            else:
                img = Image.open(selfie).convert("RGB")
                path = UPLOADS / f"selfie_{int(time.time())}.jpg"
                img.save(path)
                uid = add_user(st.session_state.first, st.session_state.last, st.session_state.email, path)
                add_booking(uid, st.session_state.flight, st.session_state.gate)
                st.success(f"Saved traveler #{uid} for flight {st.session_state.flight} (Gate {st.session_state.gate}).")
                st.image(str(path), width=220)

def ui_monitor():
    st.subheader("🎥 Pod Monitoring (Upload short clip)")
    sound_controls(); voice_test_widget()

    if not MP_AVAILABLE:
        st.error("MediaPipe not available on this deployment.")
        return

    users = get_users()
    if not users:
        st.info("No users yet. Please check-in first.")
        return

    ulabel = {f"{u['first_name']} {u['last_name']} (#{u['id']})": u["id"] for u in users}
    uid = ulabel[st.selectbox("Select traveler", list(ulabel.keys()))]

    use_yolo = st.checkbox("YOLO person overlay (optional, needs ultralytics)", value=False)

    up = st.file_uploader("Upload MP4/MOV", type=["mp4","mov","m4v"])
    if up:
        tmp = UPLOADS / f"video_{uid}_{int(time.time())}.mp4"
        with open(tmp, "wb") as f: f.write(up.read())
        st.video(str(tmp))
        if st.button("Analyze"):
            with st.spinner("Analyzing..."):
                out_path, summary = process_video(tmp, use_yolo=use_yolo)
            st.success("Done.")
            st.video(str(out_path))
            c = st.columns(3)
            c[0].metric("Final", summary["final_status"])
            c[1].metric("Max Sleep Streak (s)", f"{summary['max_consec_sleep_sec']:.1f}")
            c[2].metric("Sleepy %", f"{summary['sleepy_ratio']*100:.1f}%")
            flights_df = load_flights()
            mins = minutes_to_departure_for_user(uid, flights_df)
            if summary["final_status"]=="SLEEPY" and within_t30(mins):
                st.error(f"🚨 WAKE UP! Flight in {mins:.1f} min.")
                wake_sound()
            elif summary["final_status"]=="SLEEPY":
                st.warning("SLEEPY detected, but not inside T-30 window.")
            else:
                st.info("AWAKE — no alert.")
            log_status(uid, summary["final_status"], info=json.dumps(summary))
    else:
        st.info("Upload a short clip to begin.")

def ui_alerts(flights_df):
    st.subheader("⏰ Alerts (≤30 min to departure & traveler is SLEEPY)")
    if st.button("Refresh"):
        safe_rerun()
    st.write(f"**Now (Asia/Dubai):** {fmt_ts(now_tz())}")
    alerts = compute_alerts(flights_df)
    if alerts:
        for a in alerts:
            st.error(f"**Wake up {a['user']}!** Flight **{a['flight_no']}** (Gate **{a['gate']}**) "
                     f"in **{a['minutes']} min**. Last status: **{a['last']['status']}** at {a['last']['ts']}.")
    else:
        st.success("No alerts right now.")

# ----------------- Live Monitor (WebRTC – works on Streamlit Cloud) ------------
def ui_live_monitor():
    st.subheader("🟢 Live Webcam Monitor (EAR-based sleep detection)")
    sound_controls(); voice_test_widget()

    if not MP_AVAILABLE:
        st.error("MediaPipe not available on this deployment.")
        return

    users = get_users()
    if not users:
        st.info("No users yet. Please check-in first.")
        return

    ulabel = {f"{u['first_name']} {u['last_name']} (#{u['id']})": u["id"] for u in users}
    uid = ulabel[st.selectbox("Traveler", list(ulabel.keys()), key="live_sel")]

    c1, c2, c3 = st.columns(3)
    respect_t30 = c3.toggle("Respect T-30 window", value=True,
                            help="Only alert inside 0–30 min to departure. Turn off for testing.")
    ear_thresh = st.slider("Eye closure threshold (EAR)", 0.15, 0.35, 0.24, 0.01)
    min_streak = st.slider("Trigger if eyes closed for (seconds)", 1.0, 5.0, 2.0, 0.5)

    status_slot = st.empty()
    window_slot = st.empty()
    alert_slot = st.empty()

    sget("alert_active", False)
    sget("awake_streak_frames", 0)

    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    state = {"consec_closed": 0, "fps": 20.0, "last_t": time.time()}

    def ear_from_lm(lm, w, h, idxs):
        pts = [(int(lm[j].x*w), int(lm[j].y*h)) for j in idxs]
        def d(a,b): return math.hypot(a[0]-b[0], a[1]-b[1])
        ear = (d(pts[1],pts[5]) + d(pts[2],pts[4])) / (2.0*d(pts[0],pts[3]) + 1e-6)
        return ear, pts

    class Processor:
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            now = time.time()
            dt_s = max(1e-3, now - state["last_t"])
            state["fps"] = 0.9*state["fps"] + 0.1*(1.0/dt_s)
            state["last_t"] = now

            status = "AWAKE"; ear_val = None

            if res.multi_face_landmarks:
                H, W, _ = img.shape
                lm = res.multi_face_landmarks[0].landmark
                ear_l, L = ear_from_lm(lm, W, H, LEFT_EYE)
                ear_r, R = ear_from_lm(lm, W, H, RIGHT_EYE)
                ear_val = (ear_l + ear_r) / 2.0
                for (x,y) in L+R: cv2.circle(img,(x,y),2,(0,255,0),-1)
                if ear_val < ear_thresh:
                    status = "SLEEPY"; state["consec_closed"] += 1
                else:
                    state["consec_closed"] = 0

            cv2.putText(img, f"Status: {status}" + (f" | EAR={ear_val:.2f}" if ear_val else ""),
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3)
            cv2.putText(img, f"Status: {status}" + (f" | EAR={ear_val:.2f}" if ear_val else ""),
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            flights_df = load_flights()
            mins = minutes_to_departure_for_user(uid, flights_df)
            if mins is None:
                window_slot.warning("No booking / flight time found for this traveler.")
            else:
                window_slot.info(f"Minutes to departure: **{mins:.1f}**  |  T-30 window: **{within_t30(mins)}**")

            need_frames = int(min_streak * max(8.0, state["fps"]))
            status_slot.write(f"Closed-eye frames: **{state['consec_closed']}** / {need_frames}  |  FPS≈{state['fps']:.1f}")

            should_alert = state["consec_closed"] >= need_frames
            if respect_t30:
                should_alert = should_alert and within_t30(mins)

            if should_alert and not st.session_state.alert_active:
                msg = "🚨 WAKE UP!"
                if mins is not None: msg += f" Flight in {mins:.1f} min."
                alert_slot.error(msg); wake_sound()
                st.session_state.alert_active = True
                st.session_state.awake_streak_frames = 0

            if st.session_state.alert_active:
                if state["consec_closed"] == 0:  # eyes opened
                    st.session_state.awake_streak_frames += 1
                    if st.session_state.awake_streak_frames >= int(max(8.0, state["fps"]) * 1.0):
                        alert_slot.success("✅ Good luck!"); success_sound()
                        st.session_state.alert_active = False
                        st.session_state.awake_streak_frames = 0
                else:
                    st.session_state.awake_streak_frames = 0

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="smartpod-live",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=Processor,
        media_stream_constraints={"video": True, "audio": False},
    )

# =============================================================================
# Main
# =============================================================================
def main():
    st.set_page_config(page_title="SmartPod Demo", page_icon="😴", layout="wide")
    st.title("😴✈️ SmartPod Demo – Sleep Pod Monitoring + Flight Alerts")
    st.caption("Flight board, traveler check-in with selfie, uploaded video analysis, LIVE webcam via WebRTC with T-30 logic + browser audio & voice alerts, and T-30 departure alerts.")

    init_db()
    flights_df = load_flights()

    tabs = st.tabs(["Flight Board", "Pod Check-In", "Pod Monitor", "Alerts", "Live Monitor"])
    with tabs[0]: ui_board(flights_df)
    with tabs[1]: ui_checkin(flights_df)
    with tabs[2]: ui_monitor()
    with tabs[3]: ui_alerts(flights_df)
    with tabs[4]: ui_live_monitor()

    st.sidebar.markdown("### Tips")
    st.sidebar.write("• Click **🔊 Enable Sound** once per tab to unlock audio/voice.")
    st.sidebar.write("• Turn OFF **Respect T-30 window** to test alerts immediately.")
    st.sidebar.write("• Alert auto-clears when you’re awake ~1s (says “Good luck!”).")

if __name__ == "__main__":
    main()

