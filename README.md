## RetailVision — Real-Time Footfall Analytics Using Computer Vision

RetailVision is an AI-powered application that uses real-time computer vision to detect and count people in retail spaces.
It helps in understanding customer flow, analyzing store traffic patterns, and generating meaningful insights using live camera feeds or uploaded videos.

### 🚀 Overview
RetailVision provides an interactive interface where users can:<br>
👁️ Run real-time footfall detection<br>
🎥 Upload videos for analysis<br>
🎬 Use a built-in demo video<br>
📉 View summarized analytics and reports<br>
📊 Explore insights in a dashboard<br>

Built using YOLO models, Streamlit, and OpenCV, the tool is lightweight and easy to deploy.

## Key Features
### 🔴 Live Analysis
- Uses your device camera
- Detects people in real-time
- Live count overlay
- Reset & stop functionality

🎞 Video Upload

Upload MP4 videos

Frame-by-frame processing

Generates summary statistics

🎬 Demo Mode

One-click demo option

Preloaded sample video

Ideal for quick presentations

📊 Dashboard

Shows cumulative stats

Footfall visualization

Simple, clean analytics

🏗️ Tech Stack
Component	Technology
UI Framework	Streamlit
Computer Vision	YOLO + OpenCV
Programming Language	Python
Visualization	Streamlit Charts
Deployment	Streamlit Cloud / Local
📁 Project Structure
RetailVision/
│── app.py
│── models/
│   └── best.pt
│── media/
│   └── demo.mp4
│── .streamlit/
│   └── static/
│       └── header_bg.jpg
│── requirements.txt
│── README.md

⚙️ Setup Instructions
1. Clone the Repository
