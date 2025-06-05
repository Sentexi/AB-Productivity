# Task Analytics

Python utilities for analysing task throughput and efficiency.

## Usage

Place your CSV exports inside a dated folder under `data/`, e.g. `data/04_Jun_2025/my_export_all.csv`.

To generate static graphs via the command line:

```bash
python src/report.py 04_Jun_2025
```

### Web Application

A simple Flask web app is available for interactive exploration. On startup it will
fetch the latest data from Notion if a folder for the current day does not
already exist. The app then loads the most recent dated folder in `data/` and
builds an SQLite database along with all graphs.

Run it with:

```bash
python app/app.py
```

Open `http://localhost:5000` in your browser to view the dashboard.

All graphs are saved under `app/static/img`.
