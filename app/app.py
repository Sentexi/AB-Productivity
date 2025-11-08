import os
import glob
import sqlite3
import sys
from datetime import datetime, timedelta
from typing import Optional, Tuple

from flask import Flask, render_template, request
import plotly.offline as pyo
import plotly.graph_objects as go
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.database import export_database_csv
from src.report import (
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
    interactive_ttc_histogram,
    interactive_monthly_task_flow,
    interactive_weekly_time_minutes,
    interactive_weekly_task_flow_counts,
    interactive_workspace_piecharts,
    interactive_daily_time_backlog,
    interactive_waterfall,
    prepare_weekly_time_minutes,
    prepare_daily_time_backlog,
    prepare_weekly_task_flow_counts,
    prepare_ttc_statistics,
)
app = Flask(__name__, static_folder="static", template_folder="templates")

DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
IMG_DIR = os.path.join(BASE_DIR, 'static', 'img')
os.makedirs(IMG_DIR, exist_ok=True)

df_global = None
analysis_global = None

TIME_RANGE_OPTIONS = {
    "last-4-weeks": "Last 4 weeks",
    "quarter-to-date": "Quarter to Date",
    "year-to-date": "Year to Date",
}
DEFAULT_TIME_RANGE = "last-4-weeks"


def _ensure_today_folder() -> None:
    """Fetch data from Notion if today's folder is missing."""
    today = datetime.now().strftime('%d_%b_%Y')
    folder_path = os.path.join(DATA_DIR, today)
    csv_files = glob.glob(os.path.join(folder_path, '*_all.csv'))
    if not csv_files:
        export_database_csv(folder_path)


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
    _ensure_today_folder()
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


def _get_timeframe_bounds(range_key: str) -> Tuple[Optional[pd.Timestamp], pd.Timestamp]:
    """Return the (start, end) bounds for the selected dashboard range."""

    today = pd.Timestamp.today().normalize()
    if range_key == "last-4-weeks":
        return today - pd.Timedelta(weeks=4), today
    if range_key == "quarter-to-date":
        quarter = today.quarter
        start_month = 3 * (quarter - 1) + 1
        start = pd.Timestamp(year=today.year, month=start_month, day=1)
        return start, today
    if range_key == "year-to-date":
        start = pd.Timestamp(year=today.year, month=1, day=1)
        return start, today
    return None, today


def _as_naive(series: pd.Series) -> pd.Series:
    converted = pd.to_datetime(series, errors="coerce")
    if pd.api.types.is_datetime64tz_dtype(converted.dtype):
        converted = converted.dt.tz_localize(None)
    return converted


def _filter_dataframe_for_timeframe(
    df: pd.DataFrame, start_date: Optional[pd.Timestamp]
) -> pd.DataFrame:
    if start_date is None or df.empty:
        return df

    data = df.copy()
    mask = pd.Series(False, index=data.index)

    if "Created time" in data.columns:
        created = _as_naive(data["Created time"])
        data["Created time"] = created
        mask = mask | (created >= start_date)

    if "Last edited" in data.columns:
        edited = _as_naive(data["Last edited"])
        data["Last edited"] = edited
        mask = mask | (edited >= start_date)

    return data.loc[mask].copy()


def _trim_timeframe(
    df: pd.DataFrame, column: str, start_date: Optional[pd.Timestamp]
) -> pd.DataFrame:
    if start_date is None or df.empty or column not in df.columns:
        return df

    values = _as_naive(df[column])
    trimmed = df.loc[values >= start_date].copy()
    return trimmed


def _latest_activity_date(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    if df.empty:
        return None

    dates = []
    for column in ("Created time", "Last edited"):
        if column in df.columns:
            series = _as_naive(df[column])
            if not series.dropna().empty:
                dates.append(series.max())
    if not dates:
        return None
    return max(dates)


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


# -------- Interactive routes --------

@app.route('/interactive')
def interactive_index():
    if df_global is None:
        _load_data()
    return render_template('interactive/index.html')


@app.route('/dashboard')
def dashboard():
    if df_global is None or analysis_global is None:
        _load_data()

    selected_range = request.args.get('range', DEFAULT_TIME_RANGE)
    if selected_range not in TIME_RANGE_OPTIONS:
        selected_range = DEFAULT_TIME_RANGE

    start_date, default_end = _get_timeframe_bounds(selected_range)
    df_filtered = _filter_dataframe_for_timeframe(df_global.copy(), start_date)

    weekly_minutes = _trim_timeframe(
        prepare_weekly_time_minutes(df_filtered), 'week_start', start_date
    )
    weekly_counts = _trim_timeframe(
        prepare_weekly_task_flow_counts(df_filtered), 'week_start', start_date
    )
    daily_backlog = _trim_timeframe(
        prepare_daily_time_backlog(df_filtered), 'date', start_date
    )
    ttc_stats, ttc_done = prepare_ttc_statistics(df_filtered)

    if start_date is not None and not ttc_done.empty:
        last_edited = _as_naive(ttc_done['Last edited'])
        mask = last_edited >= start_date
        ttc_done = ttc_done.loc[mask].copy()
        ttc_done['Last edited'] = last_edited.loc[mask]

    latest_activity = _latest_activity_date(df_filtered)
    end_date = latest_activity or default_end
    range_label = TIME_RANGE_OPTIONS[selected_range]
    range_summary = (
        f"{start_date.strftime('%b %d, %Y')} – {end_date.strftime('%b %d, %Y')}"
        if start_date is not None
        else "All available data"
    )

    kpi_cards = []

    # Weekly Throughput
    if not weekly_counts.empty:
        weekly_counts_sorted = weekly_counts.sort_values('week_start')
        latest_week = weekly_counts_sorted.iloc[-1]
        prev_week = weekly_counts_sorted.iloc[-2] if len(weekly_counts_sorted) > 1 else None
        delta_done = (
            latest_week['tasks_done'] - prev_week['tasks_done']
            if prev_week is not None
            else None
        )
        spark_weeks = weekly_counts_sorted.tail(8)
        spark_div = _sparkline_div(
            spark_weeks['label'],
            spark_weeks['tasks_done'] - spark_weeks['tasks_created'],
            color="#198754",
        )
        kpi_cards.append(
            {
                'title': 'Weekly Throughput',
                'primary': f"{int(latest_week['tasks_done'])} done",
                'secondary': f"{int(latest_week['tasks_created'])} created",
                'delta': _format_delta(delta_done, unit=" vs prior"),
                'sparkline': spark_div,
                'footer': f"Week of {latest_week['label']}",
            }
        )

    # Time Investment
    if not weekly_minutes.empty:
        weekly_minutes_sorted = weekly_minutes.sort_values('week_start')
        latest_time = weekly_minutes_sorted.iloc[-1]
        variance = latest_time['actual_minutes'] - latest_time['estimated_minutes']
        spark_time = weekly_minutes_sorted.tail(8)
        spark_div = _sparkline_div(
            spark_time['label'],
            spark_time['actual_minutes'],
            color="#0d6efd",
        )
        delta = _format_delta(variance, unit=" min", invert=True)
        kpi_cards.append(
            {
                'title': 'Time Investment',
                'primary': f"{int(latest_time['actual_minutes']):,} min actual",
                'secondary': f"{int(latest_time['estimated_minutes']):,} min planned",
                'delta': delta,
                'sparkline': spark_div,
                'footer': 'Variance vs plan (lower is better)',
            }
        )

    # Backlog Snapshot
    if not daily_backlog.empty:
        daily_sorted = daily_backlog.sort_values('date')
        latest_day = daily_sorted.iloc[-1]
        prev_day = daily_sorted.iloc[-2] if len(daily_sorted) > 1 else None
        delta_backlog = (
            latest_day['cumulative_backlog'] - prev_day['cumulative_backlog']
            if prev_day is not None
            else None
        )
        spark_daily = daily_sorted.tail(14)
        spark_div = _sparkline_div(
            spark_daily['date'],
            spark_daily['cumulative_backlog'],
            color="#6610f2",
        )
        kpi_cards.append(
            {
                'title': 'Backlog Snapshot',
                'primary': _format_number(latest_day['cumulative_backlog'], unit=' min'),
                'secondary': f"Net {int(latest_day['net_change']):,} min last day",
                'delta': _format_delta(delta_backlog, unit=' min', invert=True),
                'sparkline': spark_div,
                'footer': f"Updated {latest_day['date']:%b %d, %Y}",
            }
        )

    # Median TTC
    stats = ttc_stats
    recent_median = stats.get('recent_median')
    overall_median = stats.get('overall_median')
    if recent_median is not None or overall_median is not None:
        delta_value = None
        if recent_median is not None and overall_median is not None:
            delta_value = recent_median - overall_median
        recent_series = (
            ttc_done.sort_values('Last edited').tail(30)
            if not ttc_done.empty
            else pd.DataFrame(columns=['Last edited', 'TTC'])
        )
        spark_div = _sparkline_div(
            recent_series['Last edited'],
            recent_series['TTC'],
            color="#fd7e14",
        )
        kpi_cards.append(
            {
                'title': 'Median TTC',
                'primary': _format_number(recent_median, unit=' days'),
                'secondary': f"Overall { _format_number(overall_median, unit=' days') }",
                'delta': _format_delta(delta_value, unit=' days', invert=True),
                'sparkline': spark_div,
                'footer': f"{stats.get('recent_count', 0)} tasks completed in window",
            }
        )

    # Workspace Focus (align with selected window)
    if not df_filtered.empty:
        df_workspace = df_filtered.copy()
        df_workspace['Created time'] = _as_naive(df_workspace['Created time'])
        cutoff_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=30)
        if start_date is not None:
            cutoff_date = max(cutoff_date, start_date)
        recent_workspace = df_workspace[df_workspace['Created time'] >= cutoff_date]
        if recent_workspace.empty:
            recent_workspace = df_workspace
        workspace_counts = (
            recent_workspace.groupby('Workspace').size().sort_values(ascending=False)
        )
        if not workspace_counts.empty:
            top_workspace_raw = workspace_counts.index[0]
            top_workspace = (
                str(top_workspace_raw)
                if pd.notna(top_workspace_raw)
                else 'Unspecified'
            )
            total_recent = workspace_counts.sum()
            share = workspace_counts.iloc[0] / total_recent * 100 if total_recent else 0
            spark_div = None
            if workspace_counts.shape[0] > 1:
                share_series = (
                    recent_workspace
                    .set_index('Created time')
                    .groupby(pd.Grouper(freq='W'))['Workspace']
                    .apply(lambda x: (x == top_workspace).mean() * 100)
                    .dropna()
                )
                if not share_series.empty:
                    spark_div = _sparkline_div(
                        share_series.index.strftime('%Y-%m-%d'),
                        share_series.values,
                        color="#20c997",
                    )
            kpi_cards.append(
                {
                    'title': 'Workspace Focus',
                    'primary': f"{top_workspace}",
                    'secondary': f"{share:.1f}% of {range_label.lower()}",
                    'delta': {'text': '', 'is_positive': True},
                    'sparkline': spark_div,
                    'footer': 'Share of tasks by workspace',
                }
            )

    # Flow Streak
    if not daily_backlog.empty:
        streak = 0
        for change in reversed(daily_backlog.sort_values('date')['net_change']):
            if change <= 0:
                streak += 1
            else:
                break
        spark_div = _sparkline_div(
            daily_backlog.tail(14)['date'],
            daily_backlog.tail(14)['net_change'],
            color="#dc3545",
        )
        kpi_cards.append(
            {
                'title': 'Flow Streak',
                'primary': f"{streak} day streak",
                'secondary': 'Completions ≥ inflow',
                'delta': _format_delta(
                    daily_backlog.iloc[-1]['net_change'],
                    unit=' min net',
                    invert=True,
                ),
                'sparkline': spark_div,
                'footer': 'Recent daily net changes',
            }
        )

    # Trend charts for dashboard sections
    weekly_counts_fig = interactive_weekly_task_flow_counts(
        df_filtered, start_date=start_date
    )
    weekly_counts_div = _fig_to_div(weekly_counts_fig)

    weekly_minutes_fig = interactive_weekly_time_minutes(
        df_filtered, start_date=start_date
    )
    weekly_minutes_div = _fig_to_div(weekly_minutes_fig)

    backlog_fig = interactive_daily_time_backlog(df_filtered, start_date=start_date)
    backlog_div = _fig_to_div(backlog_fig)

    ttc_fig = interactive_ttc_histogram(df_filtered, start_date=start_date)
    ttc_div = _fig_to_div(ttc_fig)

    workspace_fig = interactive_workspace_piecharts(
        df_filtered, start_date=start_date
    )
    workspace_div = _fig_to_div(workspace_fig)

    waterfall_fig = interactive_waterfall(df_filtered, start_date=start_date)
    waterfall_div = _fig_to_div(waterfall_fig)

    return render_template(
        'dashboard.html',
        kpi_cards=kpi_cards,
        weekly_counts_div=weekly_counts_div,
        weekly_minutes_div=weekly_minutes_div,
        backlog_div=backlog_div,
        ttc_div=ttc_div,
        workspace_div=workspace_div,
        waterfall_div=waterfall_div,
        range_options=TIME_RANGE_OPTIONS,
        selected_range=selected_range,
        range_label=range_label,
        range_summary=range_summary,
    )


def _fig_to_div(fig):
    return pyo.plot(fig, output_type='div', include_plotlyjs='cdn')


def _format_number(value, unit=""):
    if value is None:
        return "—"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    formatted = f"{value:,.0f}" if isinstance(value, int) else f"{value:,.1f}"
    return f"{formatted}{unit}"


def _format_delta(value, unit="", invert=False):
    if value is None:
        return {"text": "", "is_positive": False}
    sign = "" if value < 0 else "+"
    text = f"{sign}{value:,.1f}{unit}" if isinstance(value, float) else f"{sign}{int(value)}{unit}"
    is_positive = value >= 0
    if invert:
        is_positive = not is_positive
    return {"text": text, "is_positive": is_positive}


def _sparkline_div(x_values, y_values, color="#0d6efd"):
    if len(x_values) == 0:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines",
            line=dict(color=color, width=2),
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=120,
    )
    return _fig_to_div(fig)


@app.route('/interactive/ttc')
def interactive_ttc():
    if df_global is None:
        _load_data()
    fig = interactive_ttc_histogram(df_global)
    div = _fig_to_div(fig)
    return render_template('interactive/ttc.html', plot_div=div)


@app.route('/interactive/taskflow')
def interactive_taskflow():
    if analysis_global is None or df_global is None:
        _load_data()
    fig = interactive_monthly_task_flow(analysis_global)
    div = _fig_to_div(fig)
    return render_template('interactive/taskflow.html', plot_div=div)


@app.route('/interactive/time-minutes')
def interactive_time_minutes():
    if df_global is None:
        _load_data()
    weekly_fig = interactive_weekly_time_minutes(df_global)
    backlog_fig = interactive_daily_time_backlog(df_global)
    weekly_div = _fig_to_div(weekly_fig)
    backlog_div = _fig_to_div(backlog_fig)
    return render_template(
        'interactive/time_minutes.html',
        weekly_div=weekly_div,
        backlog_div=backlog_div,
    )


@app.route('/interactive/taskflow-weekly')
def interactive_taskflow_weekly():
    if df_global is None:
        _load_data()
    fig = interactive_weekly_task_flow_counts(df_global)
    div = _fig_to_div(fig)
    return render_template('interactive/taskflow_weekly.html', plot_div=div)


@app.route('/interactive/workspace')
def interactive_workspace():
    if df_global is None:
        _load_data()
    fig = interactive_workspace_piecharts(df_global)
    div = _fig_to_div(fig)
    return render_template('interactive/workspace.html', plot_div=div)


@app.route('/interactive/waterfall')
def interactive_waterfall_route():
    if df_global is None:
        _load_data()
    fig = interactive_waterfall(df_global)
    div = _fig_to_div(fig)
    return render_template('interactive/waterfall.html', plot_div=div)


if __name__ == '__main__':
    _load_data()
    app.run(debug=True)
