# CrowdGuard

CrowdGuard is a Streamlit dashboard for reviewing crowd movement and early-warning risk indicators from video. It divides a camera view into nine configurable zones and presents occupancy, movement, counter-flow, risk trends, alerts, and an annotated video to assist human operators.

The project includes a synthetic demo mode and a video-processing pipeline powered by YOLO object detection and tracking. It is an operator-support tool, not a certain prediction or a replacement for human judgment.

## Features

- Dashboard with zone risk map, people counts, average speed, counter-flow, risk trends, and alert feed.
- Demo data mode for exploring the dashboard without a video.
- Upload and process MP4, MOV, AVI, or MKV videos from the sidebar.
- Configure zone names, capacities, risk thresholds, and detection settings in `config.json` and `crowdguard/config.json`.
- Generate an annotated video, per-second timeline, alerts, forecast, and evidence frames.

## Requirements

- Python 3.10 or later
- pip
- A supported video file to run the real processing pipeline (optional for demo mode)

The first video run may download the YOLOv8 model weights configured in `crowdguard/config.json`. A network connection is needed if those weights are not already available locally.

## Run locally (Windows PowerShell)

From the project directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit prints a local URL, usually <http://localhost:8501>. Open it in your browser. If PowerShell blocks virtual-environment activation, run the commands using the virtual environment's interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Using the dashboard

1. Select **Demo data** in the sidebar to explore the dashboard without input files.
2. To analyze a video, upload it in the sidebar and select **Run pipeline**.
3. When processing finishes, the dashboard switches to real data. Use the playback controls to inspect different points in the video and the alert feed to review detected events.
4. Adjust zone capacities in the sidebar or update the configuration files for persistent settings.

## Outputs

Video processing writes generated files under `output/`:

- `timeline.csv` — per-second, per-zone measurements and risk scores.
- `alerts.json` — alerts, explanations, metrics, and evidence references.
- `forecast.json` — risk forecast information.
- `annotated.mp4` or `annotated.webm` — browser-ready annotated video when encoding succeeds.
- `frames/` — sampled evidence images.

Generated outputs, uploaded source videos, and model weights are excluded from Git by `.gitignore`. Keep a separate backup of any generated results you need to retain.

## Run the pipeline from the command line

The same video-processing pipeline can be run without the dashboard:

```powershell
python pipeline.py --help
python pipeline.py path\to\your-video.mp4
```

See `crowdguard/config.json` for the detection, tracking, zone, and risk-scoring defaults.

## Project layout

```text
app.py                  Streamlit dashboard
pipeline.py             Command-line pipeline entry point
crowdguard/             Detection, tracking, metrics, alerts, and forecasting
config.json             Dashboard zone settings
crowdguard/config.json  Pipeline and risk-scoring settings
requirements.txt        Python dependencies
```
