import os
import glob
import sqlite3
import sys
from datetime import datetime

from flask import Flask, render_template

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, '..', 'src'))

from report import (
    analyze_tasks,
    plot_time_to_completion_histogram,
    plot_monthly_task_flow,
    waterfall,
    workspace_piecharts_by_year,
    plot_median_ttc_over_time,
    plot_task_flow_time_series,
    plot_day_of_week_throughput,
    plot_working_session_heatmaps,
    plot_abbvie_done_heatmaps,
    plot_liberal_stuff_done_heatmaps,
)
app = Flask(__name__, static_folder="static", template_folder="templates")

DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
IMG_DIR = os.path.join(BASE_DIR, 'static', 'img')
os.makedirs(IMG_DIR, exist_ok=True)

df_global = None
analysis_global = None


def _latest_csv():
    folders = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))]
    if not folders:
        raise FileNotFoundError('No data folders found')

    def parse_date(name):
        try:
            return datetime.strptime(name, '%d_%b_%Y')
        except ValueError:
            return datetime.min

    latest = max(folders, key=lambda x: parse_date(x))
    folder_path = os.path.join(DATA_DIR, latest)
    csv_files = glob.glob(os.path.join(folder_path, '*_all.csv'))
    if not csv_files:
        raise FileNotFoundError('No CSV file found in latest folder')
    return csv_files[0], folder_path


def _load_data():
    global df_global, analysis_global
    csv_path, folder_path = _latest_csv()
    df, analysis = analyze_tasks(csv_path)

    db_path = os.path.join(folder_path, 'analysis.db')
    conn = sqlite3.connect(db_path)
    df.to_sql('tasks', conn, if_exists='replace', index=False)
    analysis.to_sql('analysis', conn, if_exists='replace', index=False)
    conn.close()

    # generate graphs
    plot_time_to_completion_histogram(df, os.path.join(IMG_DIR, 'TTC_All_time.png'))
    plot_monthly_task_flow(analysis, os.path.join(IMG_DIR, 'Task_flow.png'))
    waterfall(df, analysis, os.path.join(IMG_DIR, 'waterfall'))
    workspace_piecharts_by_year(df, IMG_DIR)
    plot_median_ttc_over_time(df, IMG_DIR)
    plot_task_flow_time_series(analysis, IMG_DIR)
    plot_day_of_week_throughput(df, IMG_DIR)
    plot_working_session_heatmaps(df, IMG_DIR)
    plot_abbvie_done_heatmaps(df, IMG_DIR)
    plot_liberal_stuff_done_heatmaps(df, IMG_DIR)

    df_global = df
    analysis_global = analysis


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/graphs')
def graphs():
    imgs = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.png')])
    return render_template('graphs.html', graphs=imgs)


@app.route('/analysis')
def analysis():
    if analysis_global is None:
        table_html = '<p>No data loaded.</p>'
    else:
        table_html = analysis_global.to_html(classes='data', index=False)
    return render_template('analysis.html', table=table_html)


if __name__ == '__main__':
    _load_data()
    app.run(debug=True)
