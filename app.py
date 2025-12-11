# app.py - RetailVision AI (updated: stop-demo option + reset counts on new run)
import os
import tempfile
import math
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO
import cvzone
from sort import Sort
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu

# ----------------------------- Page config -----------------------------
st.set_page_config(
    page_title="RetailVision AI | Smart Footfall",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Minimal CSS -----------------------------
st.markdown(
    """
<style>
.section-header { background: linear-gradient(90deg,#1a2980 0%,#26d0ce 100%); color: white; padding: 14px; border-radius: 10px; box-shadow: 0 6px 18px rgba(0,0,0,0.12); }
.metric-card { background: white; padding: 12px; border-radius: 10px; box-shadow: 0 6px 18px rgba(0,0,0,0.06); }
.video-container { border-radius: 10px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.12); background: #000; }
.small-muted { color: #98a1ac; font-size: 0.9rem; }
.btn-primary > button { background: linear-gradient(90deg,#667eea,#764ba2); color: white; border-radius:8px; border: none; padding: 8px 12px; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------- Sidebar (essential config) -----------------------------
with st.sidebar:
    st.markdown("<div style='text-align:center'>", unsafe_allow_html=True)
    st.markdown("<h2 style='margin:6px'>🛍️ RetailVision AI</h2>", unsafe_allow_html=True)
    st.markdown("<p class='small-muted'>Live footfall counting</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    page = option_menu(
        menu_title=None,
        options=["Live", "Dashboard"],
        icons=["play-circle", "bar-chart-line"],
        menu_icon="cast",
        default_index=0,
        styles={"nav-link-selected": {"background-color": "#667eea", "color": "white"}},
    )

    st.sidebar.divider()
    st.sidebar.markdown("### ⚙️ Settings (essential)")
    # Counting line
    line_y = st.sidebar.slider("Counting line Y (px)", 0, 1080, 360)
    line_x1 = st.sidebar.slider("Line X start", 0, 1920, 100)
    line_x2 = st.sidebar.slider("Line X end", 0, 1920, 1180)
    st.sidebar.markdown("---")

    confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.35, 0.05)
    heatmap_mode = st.sidebar.checkbox("Enable heatmap overlay", False)
    show_ids = st.sidebar.checkbox("Show IDs", True)
    show_confidence = st.sidebar.checkbox("Show confidences", False)
    st.sidebar.markdown("---")
    st.sidebar.markdown("Tip: Keep confidence ~0.3–0.5 for general scenarios.")

# ----------------------------- Model (cached) -----------------------------
@st.cache_resource
def load_model(path="yolov8n.pt"):
    return YOLO(path)

model = load_model()

# ----------------------------- Globals & Helpers -----------------------------
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

def create_heatmap_overlay(img, detections, alpha=0.35):
    heatmap = np.zeros(img.shape[:2], dtype=np.uint8)
    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(heatmap, (cx, cy), 40, 255, -1)
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)

def create_trend_figure(series, height=250):
    fig = go.Figure(go.Scatter(x=list(range(len(series))), y=series, mode="lines+markers"))
    fig.update_layout(margin=dict(l=10, r=10, t=25, b=25), height=height, plot_bgcolor="white")
    return fig

def reset_analysis_state():
    """Reset counters and logs before starting a new run"""
    st.session_state.total_ids = set()
    st.session_state.trend_data = []
    st.session_state.events_df = pd.DataFrame(columns=["timestamp", "cumulative_count"])
    st.session_state.analysis_start_time = datetime.now()

# ----------------------------- Session state initialization -----------------------------
if "total_ids" not in st.session_state:
    st.session_state.total_ids = set()
if "trend_data" not in st.session_state:
    st.session_state.trend_data = []
if "events_df" not in st.session_state:
    st.session_state.events_df = pd.DataFrame(columns=["timestamp", "cumulative_count"])
if "analyzing" not in st.session_state:
    st.session_state.analyzing = False
if "analysis_start_time" not in st.session_state:
    st.session_state.analysis_start_time = None
if "current_run_label" not in st.session_state:
    st.session_state.current_run_label = None  # "demo", "live", "upload", "stream" or None

# ----------------------------- Video processing -----------------------------
DISPLAY_WIDTH = 900  # change if you want a different display width (pixels)

def process_video_capture(cap, input_label="file"):
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    limits = [line_x1, line_y, line_x2, line_y]

    frame_count = 0
    progress = st.session_state.get("_prog", None)
    if progress is None:
        progress = st.progress(0)

    video_col = st.empty()

    # mark current run label
    st.session_state.current_run_label = input_label

    while True:
        # If user clicked Stop (anywhere) or switched runs, break
        if not st.session_state.analyzing:
            break

        success, frame = cap.read()
        if not success:
            if input_label not in ["Live Camera", "stream"]:
                # loop files so demo videos keep showing
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                st.warning("Live camera disconnected or stream ended.")
                break

        frame_count += 1

        # YOLO inference
        results = model(frame, stream=True, verbose=False)
        detections = np.empty((0, 5))

        for r in results:
            boxes = getattr(r, "boxes", [])
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if classNames[cls] == "person" and conf >= confidence_threshold:
                    detections = np.vstack((detections, np.array([x1, y1, x2, y2, conf])))

        tracked = tracker.update(detections)

        # Optional heatmap
        if heatmap_mode and len(detections) > 0:
            frame = create_heatmap_overlay(frame, detections)

        # Draw counting line
        cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 4)

        # Process tracked boxes
        for t in tracked:
            x1, y1, x2, y2, tid = t
            x1, y1, x2, y2, tid = int(x1), int(y1), int(x2), int(y2), int(tid)
            w, h = x2 - x1, y2 - y1

            # fancy rect & label
            cvzone.cornerRect(frame, (x1, y1, w, h), l=8, rt=3)
            label = f"ID:{tid}"
            if show_ids:
                cvzone.putTextRect(frame, label, (x1, max(20, y1)), scale=0.7, thickness=1, offset=5)

            # center
            cx, cy = x1 + w // 2, y1 + h // 2
            if limits[0] < cx < limits[2] and limits[1] - 12 < cy < limits[1] + 12:
                if tid not in st.session_state.total_ids:
                    st.session_state.total_ids.add(tid)
                    cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 200, 0), 6)

        # update trend every N frames (approx per second depending on FPS)
        if frame_count % 30 == 0:
            st.session_state.trend_data.append(len(st.session_state.total_ids))
            ts = datetime.now()
            st.session_state.events_df = pd.concat(
                [st.session_state.events_df,
                 pd.DataFrame([{"timestamp": ts, "cumulative_count": len(st.session_state.total_ids)}])],
                ignore_index=True,
            )

        # overlay total & timestamp
        txt = f"TOTAL: {len(st.session_state.total_ids)}"
        cvzone.putTextRect(frame, txt, (20, 40), scale=1.7, thickness=2, offset=8, colorT=(255, 255, 255))
        cv2.putText(frame, datetime.now().strftime("%H:%M:%S"), (frame.shape[1] - 180, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # show frame — use explicit width to avoid deprecation warning
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_col.image(frame_rgb, width=DISPLAY_WIDTH)

        # progress update for files
        if input_label not in ["Live Camera", "stream"]:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
            prog_val = min(frame_count / total, 1.0)
            progress.progress(prog_val)

    try:
        cap.release()
    except:
        pass
    progress.empty()
    # clear current run label when finished/stopped
    st.session_state.current_run_label = None


# ----------------------------- UI: Live page -----------------------------
if page == "Live":
    # --- 1. CSS TO REMOVE PADDING & FIX OVERLAY ---
    st.markdown("""
    <style>
        /* override Streamlit's default container padding */
        .main .block-container {
            padding-top: 1rem !important; /* Keep a tiny bit space at top */
            padding-left: 0 !important;
            padding-right: 0 !important;
            max-width: 100% !important;
        }

        /* Hero container styles */
        .hero-wrap {
            position: relative;
            width: 100vw; /* Force full viewport width */
            height: 280px; /* Slightly taller */
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0; /* Remove margin */
            left: 50%;
            right: 50%;
            margin-left: -50vw; /* Hack to center full-width element */
            margin-right: -50vw;
        }
        
        /* Background image styles */
        .hero-wrap img.bg-img {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover; /* Ensures image covers area without distorting */
            z-index: 0;
        }
        
        /* Dark Overlay - Made darker for better text visibility */
        .hero-wrap .overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7); /* Increased opacity to 0.7 */
            z-index: 1;
        }
        
        /* Content text styles */
        .hero-content {
            position: relative;
            z-index: 2;
            text-align: center;
            color: #fff;
            max-width: 1100px;
            padding: 20px;
            font-family: 'Montserrat', system-ui, -apple-system, sans-serif;
        }
        .hero-title {
            margin: 0;
            font-size: 3rem;
            font-weight: 800;
            letter-spacing: 1px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        .hero-sub {
            margin-top: 15px;
            color: rgba(255,255,255,0.95);
            font-size: 1.1rem;
            line-height: 1.5;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
        }
    </style>
    """, unsafe_allow_html=True)

    # --- 2. IMAGE LOADER HELPER ---
    import base64
    def get_img_as_base64(file_path):
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()

    # Load local image or fallback
    img_path = "header.jpg" 
    if os.path.exists(img_path):
        img_b64 = get_img_as_base64(img_path)
        header_img_src = f"data:image/jpg;base64,{img_b64}"
    else:
        # Professional retail fallback image
        header_img_src = "https://images.unsplash.com/photo-1556742049-0c63d7cb71f7?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80"

    # --- 3. THE HTML HERO SECTION ---
    st.markdown(
        f"""
        <div class="hero-wrap" role="banner">
          <img class="bg-img" src="{header_img_src}" alt="retail background" />
          <div class="overlay"></div>
          <div class="hero-content">
            <h1 class="hero-title">🛍️ RetailVision — Live Analytics</h1>
            <p class="hero-sub">
              Real-time footfall analytics for retail environments. Detect, track, and count visitors 
              with AI precision. Visualize trends and gain actionable in-store insights.
            </p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True) # Add some space below

    # ... (Keep the rest of your 'How to use' and demo logic here) ...


    st.markdown("")  # spacing

    # How to use + Demo
    with st.expander("How to use (quick) — demo available", expanded=True):
        st.markdown("""
        **Quick steps**
        1. Pick an input (Live Camera / Upload / Stream URL).  
        2. Click Start. The app will detect people, assign IDs and count when they cross the red line.  
        3. Click **Generate Report** to get a concise CSV and summary.
        """)
        demo_col1, demo_col2 = st.columns([2, 1])
        with demo_col1:
            st.markdown("**Use demo to understand**: click the button to load a pre-trained sample video (`people.mp4`).")
            if os.path.exists("people.mp4"):
                st.video("people.mp4", start_time=0)
            else:
                st.info("Demo file `people.mp4` not found in the app folder. Place a sample video named `people.mp4` to enable instant demo.")
        with demo_col2:
            # Show Play or Stop depending on current run
            if st.session_state.current_run_label == "demo" and st.session_state.analyzing:
                if st.button("⏹ Stop Demo"):
                    # stop and clear analyzing state
                    st.session_state.analyzing = False
                    st.session_state.current_run_label = None
                    st.success("Demo stopped.")
            else:
                if st.button("▶️ Play Demo Video (people.mp4)"):
                    if os.path.exists("people.mp4"):
                        # reset counts for a fresh demo run
                        reset_analysis_state()
                        st.session_state.analyzing = True
                        st.session_state.analysis_start_time = datetime.now()
                        cap = cv2.VideoCapture("people.mp4")
                        process_video_capture(cap, "demo")
                        st.session_state.analyzing = False
                    else:
                        st.error("Demo video 'people.mp4' not found in application folder. Upload it or place people.mp4 next to app.py.")
    st.markdown("---")

    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.markdown("### Input")
        input_method = st.radio("", ["Live Camera", "Upload Video", "Stream URL"], horizontal=True, key="input_method")

        # --- Robust Live Camera start block ---
        if input_method == "Live Camera":
            if st.button("▶️ Start Live Analysis", key="start_live"):
                # Reset counts before starting new run
                reset_analysis_state()
                st.session_state.analyzing = True
                st.session_state.analysis_start_time = datetime.now()

                # Try a few backends/indexes to improve chances on different OSes
                tried = []
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(0)
                tried.append(("CAP_DSHOW", cap.isOpened()))
                if not cap.isOpened() and hasattr(cv2, "CAP_MSMF"):
                    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
                    tried.append(("CAP_MSMF", cap.isOpened()))
                if not cap.isOpened():
                    cap = cv2.VideoCapture(0)
                    tried.append(("DEFAULT", cap.isOpened()))
                # try index 1 if 0 fails (external cams)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(1)
                    tried.append(("INDEX_1", cap.isOpened()))

                attempts_text = ", ".join([f"{name}:{'OK' if ok else 'NO'}" for name, ok in tried])
                st.info(f"Camera attempts: {attempts_text}")

                if not cap.isOpened():
                    st.error(
                        "Unable to open the camera. Possible reasons:\n"
                        "- The app is not running on the local machine that has the webcam (remote server).\n"
                        "- Camera permissions are blocked or camera is used by another app.\n"
                        "- Try a different device index (0,1,2) or run locally.\n"
                        "If you want browser-based single-frame capture, use `st.camera_input()` (but it won't stream continuously)."
                    )
                    st.session_state.analyzing = False
                else:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    process_video_capture(cap, "Live Camera")
                    st.session_state.analyzing = False

        elif input_method == "Upload Video":
            uploaded_file = st.file_uploader("Upload video (mp4/avi/mkv)", type=["mp4", "avi", "mkv"])
            if uploaded_file is not None:
                st.video(uploaded_file, start_time=0)
                if st.button("▶️ Analyze Uploaded Video"):
                    # Reset counts so uploaded video starts from zero
                    reset_analysis_state()
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    tfile.write(uploaded_file.read())
                    tfile.flush()
                    st.session_state.analyzing = True
                    cap = cv2.VideoCapture(tfile.name)
                    process_video_capture(cap, "file")
                    st.session_state.analyzing = False

        else:  # stream url
            url = st.text_input("Stream URL (rtsp/http/...)", placeholder="rtsp://username:pass@ip/stream")
            if st.button("▶️ Connect & Analyze Stream") and url:
                # Reset counts for the stream run
                reset_analysis_state()
                st.session_state.analyzing = True
                st.session_state.analysis_start_time = datetime.now()
                cap = cv2.VideoCapture(url)
                if cap.isOpened():
                    process_video_capture(cap, "stream")
                else:
                    st.error("Could not open stream. Check URL and network.")
                st.session_state.analyzing = False

    with right_col:
        st.markdown("### Live Metrics")
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("👥 Total Count", len(st.session_state.total_ids))
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("")

        st.markdown("### Trend (recent)")
        if st.session_state.trend_data:
            fig = create_trend_figure(st.session_state.trend_data[-30:])
            st.plotly_chart(fig, use_container_width=True, height=250)
        else:
            st.info("No trend data yet — run an analysis to populate this chart.")

        st.markdown("---")
        st.markdown("### Controls")
        c1, c2 = st.columns([1,1])
        if c1.button("⏹ Stop Analysis"):
            st.session_state.analyzing = False
            st.session_state.current_run_label = None

        if c2.button("📄 Generate Report"):
            st.session_state.analyzing = False
            total = len(st.session_state.total_ids)
            st.success(f"Report generated — Total unique people counted: {total}")
            st.write("Summary:")
            st.write(f"- Analysis start: {(st.session_state.events_df['timestamp'].min() if not st.session_state.events_df.empty else '—')}")
            st.write(f"- Analysis end: {datetime.now()}")
            st.write(f"- Total unique IDs: {total}")
            if not st.session_state.events_df.empty:
                st.markdown("#### Time-series sample")
                st.dataframe(st.session_state.events_df.tail(20))
            csv = st.session_state.events_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download events CSV", csv, "events.csv", "text/csv")

# ----------------------------- Dashboard page -----------------------------
else:
    st.markdown("<div class='section-header'><h3 style='margin:6px'>📊 Dashboard</h3></div>", unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>Detailed views of past analysis and trend breakdowns.</div>", unsafe_allow_html=True)
    st.markdown("")

    top1, top2 = st.columns(2)
    with top1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Unique Visitors", len(st.session_state.total_ids))
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("")
        st.markdown("#### Cumulative timeline")
        if not st.session_state.events_df.empty:
            fig = px.line(st.session_state.events_df, x="timestamp", y="cumulative_count", title="Cumulative count over time")
            fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=30))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data yet. Run an analysis to populate dashboard.")

    with top2:
        st.markdown("#### Recent samples")
        st.dataframe(st.session_state.events_df.tail(50))
        st.markdown("---")
        if not st.session_state.events_df.empty:
            csv = st.session_state.events_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Export full events CSV", csv, "events_full.csv", "text/csv")

# ----------------------------- Footer -----------------------------
st.markdown("---")
f1, f2 = st.columns(2)
with f1:
    st.markdown("**RetailVision AI** — Minimal edition")
with f2:
    st.markdown("Made with ❤️ · Streamlit + YOLOv8")
