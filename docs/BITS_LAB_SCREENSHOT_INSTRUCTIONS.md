# BITS Virtual Lab screenshot instructions

The current `screenshots/bits_lab_execution.png` is genuine and should be preserved. It shows the
BITS Virtual Lab session and the app, but its metric table is below the visible area. Do not edit that
image or recreate its timestamp, usage, username, or execution details.

## Commands to run inside the BITS Virtual Lab VM

Open a terminal in the cloned repository folder and run:

```powershell
./.venv/Scripts/python.exe model/verify_reload.py
./.venv/Scripts/python.exe model/verify_integrity.py
./.venv/Scripts/streamlit.exe run app.py --server.address 127.0.0.1
```

Open `http://127.0.0.1:8501` in the browser inside the lab VM. In the app:

1. Keep the bundled `test_data.csv` active (389 rows, 2 classes).
2. Select **Compare all five models**.
3. Zoom the lab browser out only as much as needed (usually 80% or 90%).
4. Scroll until the full five-row metrics table, including all six metric columns, is readable.

## One-screen layout

- Left 30-35%: the BITS Virtual Lab Session page with the course/session context, username, current
  timestamp, actual time-used/remaining values, and RUNNING status visible.
- Right 65-70%: the remote BITS VM browser at `127.0.0.1:8501`, with the app title, active dataset
  summary, **Compare all five models** tab, and the complete metrics table visible.
- Capture one original full-screen screenshot. Do not crop out the BITS context, modify values, or
  combine multiple screenshots.

Provide that original PNG or JPG for insertion into the final PDF. Until it is supplied, the generated
submission PDF must remain labeled DRAFT.
