import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def find_all_csv(folder_path):
    # Find any file in the folder that ends with _all.csv
    csv_files = glob.glob(os.path.join(folder_path, "*_all.csv"))
    
    if not csv_files:
        print("No matching CSV file found.")
        return None
    
    return csv_files[0]  # Return the first match

def analyze_tasks(csv_path):
    # Load CSV into DataFrame
    df = pd.read_csv(csv_path)

    # Convert relevant columns to datetime
    df["Created time"] = pd.to_datetime(df["Created time"], errors="coerce")
    df["Last edited"] = pd.to_datetime(df["Last edited"], errors="coerce")

    # Create simplified date columns
    df["Created date"] = df["Created time"].dt.date
    df["Edited date"] = df["Last edited"].dt.date

    # Count created tasks per day
    created_tasks = df.groupby("Created date").size().rename("tasks created")

    # Count done tasks per day (only where Status is "Done")
    done_tasks = df[df["Status"].str.lower() == "done"].groupby("Edited date").size().rename("tasks done")

    # Combine both into a single analysis DataFrame
    analysis = pd.concat([created_tasks, done_tasks], axis=1).fillna(0).astype(int)
    analysis.index.name = "date"
    analysis = analysis.reset_index().sort_values("date")

    return df, analysis
    
def plot_time_to_completion_histogram(df, output_path="TTC_All_time.png"):
    """
    Plots a histogram of time to completion (TTC) in days for all tasks marked as 'done',
    with the top 5% longest durations grouped into a final 'overflow' bin (e.g. '100+ days').
    """
    import matplotlib.pyplot as plt

    # Filter done tasks
    done_tasks = df[df["Status"].str.lower() == "done"].copy()

    # Ensure datetime conversion
    done_tasks["Created time"] = pd.to_datetime(done_tasks["Created time"], errors="coerce")
    done_tasks["Last edited"] = pd.to_datetime(done_tasks["Last edited"], errors="coerce")

    # Compute TTC in days
    done_tasks["time_to_completion_days"] = (done_tasks["Last edited"] - done_tasks["Created time"]).dt.days

    # Drop invalid values
    ttc_days = done_tasks["time_to_completion_days"].dropna()
    ttc_days = ttc_days[ttc_days >= 0]

    # Determine 95th percentile cutoff
    cutoff = int(ttc_days.quantile(0.95))
    clipped = ttc_days.clip(upper=cutoff)

    # Label values above the cutoff as a new bin: cutoff+1
    clipped_hist_data = clipped.copy()
    clipped_hist_data[ttc_days > cutoff] = cutoff + 1

    # Build histogram data
    bins = list(range(0, cutoff + 2))  # include the overflow bin
    labels = [str(i) for i in range(0, cutoff + 1)] + [f"{cutoff}+"]

    # Plot histogram
    plt.figure(figsize=(12, 6))
    plt.hist(clipped_hist_data, bins=bins, edgecolor='black', color='skyblue', align='left', rwidth=0.9)

    # Labeling
    plt.title("Time to Completion (TTC) – Histogram (Grouped Top 5%)")
    plt.xlabel("Days to Complete Task")
    plt.ylabel("Number of Tasks")
    plt.xticks(ticks=range(0, cutoff + 2), labels=labels, rotation=90)
    plt.grid(axis='y')
    plt.tight_layout()

    # Save plot
    plt.savefig(output_path)
    plt.close()
    print(f"TTC histogram saved to '{output_path}' (cutoff at {cutoff} days)")


def plot_monthly_task_flow(analysis_df, output_path="Task_flow.png"):
    """
    Plots and saves a bar chart showing monthly task creation (negative red bars)
    and task completion (positive green bars).
    """
    import matplotlib.pyplot as plt

    # Create 'month' column from date
    analysis_df["month"] = pd.to_datetime(analysis_df["date"]).dt.to_period("M")

    # Group by month and sum tasks
    analysis_monthly = analysis_df.groupby("month")[["tasks created", "tasks done"]].sum().reset_index()
    analysis_monthly["month"] = analysis_monthly["month"].astype(str)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(analysis_monthly["month"], -analysis_monthly["tasks created"], label="Tasks Created", color="red")
    plt.bar(analysis_monthly["month"], analysis_monthly["tasks done"], label="Tasks Done", color="green")

    plt.xlabel("Month")
    plt.ylabel("Number of Tasks")
    plt.title("Monthly Task Creation and Completion")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y')

    # Save instead of show
    plt.savefig(output_path)
    plt.close()
    print(f"Task flow chart saved to '{output_path}'")

def waterfall(df_raw, analysis_df_full, output_prefix="waterfall"):
    """
    Creates waterfall charts of cumulative task load over time for:
    - all tasks
    - tasks in 'Abbvie' workspace
    - tasks in 'Liberal Stuff' workspace

    All 'done' and 'abandoned' tasks are counted as closed.
    """

    def generate_waterfall_for_subset(df_subset, label_suffix):
        # Convert times
        df_subset["Created time"] = pd.to_datetime(df_subset["Created time"], errors="coerce")
        df_subset["Last edited"] = pd.to_datetime(df_subset["Last edited"], errors="coerce")

        # Create simplified dates
        df_subset["Created date"] = df_subset["Created time"].dt.date
        df_subset["Edited date"] = df_subset["Last edited"].dt.date

        # Count created tasks
        created = df_subset.groupby("Created date").size().rename("tasks created")

        # Count closed tasks (done or abandoned)
        closed = df_subset[df_subset["Status"].str.lower().isin(["done", "abandoned"])]
        closed = closed.groupby("Edited date").size().rename("tasks closed")

        # Merge and compute net + cumulative
        analysis = pd.concat([created, closed], axis=1).fillna(0).astype(int)
        analysis.index.name = "date"
        analysis = analysis.reset_index().sort_values("date")
        analysis["net change"] = analysis["tasks created"] - analysis["tasks closed"]
        analysis["cumulative total"] = analysis["net change"].cumsum()

        # Plot
        plt.figure(figsize=(12, 6))
        colors = analysis["net change"].apply(lambda x: "green" if x < 0 else "red")
        plt.bar(analysis["date"], analysis["net change"], color=colors, label="Net Change")
        plt.plot(analysis["date"], analysis["cumulative total"], color="black", label="Cumulative Total", linewidth=2)

        plt.title(f"Task Waterfall – {label_suffix}")
        plt.xlabel("Date")
        plt.ylabel("Number of Tasks")
        plt.legend()
        plt.grid(axis='y')
        plt.tight_layout()

        filename = f"{output_prefix}_{label_suffix.lower().replace(' ', '_')}.png"
        plt.savefig(filename)
        plt.close()
        print(f"Waterfall chart saved to '{filename}'")

    # === 1. Total ===
    generate_waterfall_for_subset(df_raw.copy(), "Total")

    # === 2. Abbvie ===
    abbvie_df = df_raw[df_raw["Workspace"].str.contains("Abbvie", case=False, na=False)].copy()
    generate_waterfall_for_subset(abbvie_df, "Abbvie")

    # === 3. Liberal Stuff ===
    liberal_df = df_raw[df_raw["Workspace"].str.contains("Liberal Stuff", case=False, na=False)].copy()
    generate_waterfall_for_subset(liberal_df, "Liberal Stuff")

def workspace_piecharts_by_year(df, output_folder):
    """
    Creates a pie chart per year, showing the share of tasks created by workspace.
    Labels include both percentage and absolute number of tasks.
    Custom colors: Abbvie (blue), Liberal Stuff (yellow), Personal (green).
    Saves each chart as '[year]_task_overview.png'.
    """
    import matplotlib.pyplot as plt
    from collections import defaultdict

    # Convert created time and extract year
    df["Created time"] = pd.to_datetime(df["Created time"], errors="coerce")
    df["year"] = df["Created time"].dt.year

    # Color map for specific workspaces
    color_map = {
        "Abbvie": "blue",
        "Liberal Stuff": "gold",
        "Personal": "green"
    }

    # Group by year and workspace
    grouped = df.groupby(["year", "Workspace"]).size().rename("count").reset_index()

    for year in sorted(grouped["year"].dropna().unique()):
        year_data = grouped[grouped["year"] == year].copy()

        # Clean workspace name (remove URL)
        year_data["Workspace Clean"] = year_data["Workspace"].str.extract(r'^(.*?)(?:\s*\(.*\))?$')[0]

        # Aggregate again in case of duplicate names
        pie_data = year_data.groupby("Workspace Clean")["count"].sum().sort_values(ascending=False)

        # Color list
        colors = [color_map.get(w.strip(), None) for w in pie_data.index]

        # Label formatter
        def autopct_format(pct, all_vals):
            absolute = int(round(pct / 100. * sum(all_vals)))
            return f"{pct:.1f}%\n({absolute})"

        # Plot
        plt.figure(figsize=(8, 8))
        wedges, texts, autotexts = plt.pie(
            pie_data,
            labels=pie_data.index,
            colors=colors,
            autopct=lambda pct: autopct_format(pct, pie_data.values),
            startangle=90,
            counterclock=False,
            textprops=dict(color="black", fontsize=10)
        )

        plt.title(f"Task Distribution by Workspace – {year}")
        plt.tight_layout()

        filename = os.path.join(output_folder, f"{int(year)}_task_overview.png")
        plt.savefig(filename)
        plt.close()
        print(f"Saved: {filename}")

def plot_median_ttc_over_time(df, output_folder):
    """
    Computes and plots median Time to Completion (TTC) for:
    - each quarter (saved as 'median_TTC.png')
    - each month (saved as 'median_TTC_monthly.png')
    """
    import matplotlib.pyplot as plt

    # Filter for done tasks
    done_df = df[df["Status"].str.lower() == "done"].copy()

    # Parse datetime
    done_df["Created time"] = pd.to_datetime(done_df["Created time"], errors="coerce")
    done_df["Last edited"] = pd.to_datetime(done_df["Last edited"], errors="coerce")

    # Calculate TTC in days
    done_df["TTC"] = (done_df["Last edited"] - done_df["Created time"]).dt.days
    done_df = done_df[done_df["TTC"] >= 0]

    # === QUARTERLY MEDIAN ===
    done_df["quarter"] = done_df["Last edited"].dt.to_period("Q").astype(str)
    ttc_quarterly = done_df.groupby("quarter")["TTC"].median().reset_index()
    ttc_quarterly = ttc_quarterly.sort_values("quarter")

    plt.figure(figsize=(10, 5))
    plt.plot(ttc_quarterly["quarter"], ttc_quarterly["TTC"], marker='o', linewidth=2, color="darkblue")
    plt.title("Median Time to Completion (TTC) per Quarter")
    plt.xlabel("Quarter")
    plt.ylabel("Median TTC (days)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "median_TTC.png"))
    plt.close()
    print("Saved: median_TTC.png")

    # === MONTHLY MEDIAN ===
    done_df["month"] = done_df["Last edited"].dt.to_period("M").astype(str)
    ttc_monthly = done_df.groupby("month")["TTC"].median().reset_index()
    ttc_monthly = ttc_monthly.sort_values("month")

    plt.figure(figsize=(12, 5))
    plt.plot(ttc_monthly["month"], ttc_monthly["TTC"], marker='o', linewidth=2, color="darkgreen")
    plt.title("Median Time to Completion (TTC) per Month")
    plt.xlabel("Month")
    plt.ylabel("Median TTC (days)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "median_TTC_monthly.png"))
    plt.close()
    print("Saved: median_TTC_monthly.png")


def plot_task_flow_time_series(analysis_df, output_folder):
    """
    Creates:
    - Monthly task flow chart (Task_flow.png)
    - Weekly task flow bar chart (task_flow_weekly.png)
    - Weekly task efficiency ratio chart (task_efficiency_weekly.png)
    - Per-year weekly task flow bar charts ([year]_task_creation_weekly.png)
    """
    import os
    import matplotlib.pyplot as plt

    # Ensure datetime parsing
    analysis_df["date"] = pd.to_datetime(analysis_df["date"])

    # === MONTHLY TASK FLOW ===
    analysis_df["month"] = analysis_df["date"].dt.to_period("M")
    monthly = analysis_df.groupby("month")[["tasks created", "tasks done"]].sum().reset_index()
    monthly["month"] = monthly["month"].astype(str)

    plt.figure(figsize=(12, 6))
    plt.bar(monthly["month"], -monthly["tasks created"], label="Tasks Created", color="red")
    plt.bar(monthly["month"], monthly["tasks done"], label="Tasks Done", color="green")
    plt.xlabel("Month")
    plt.ylabel("Number of Tasks")
    plt.title("Monthly Task Creation and Completion")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y')
    plt.savefig(os.path.join(output_folder, "Task_flow.png"))
    plt.close()
    print("Saved: Task_flow.png")

    # === WEEKLY TASK FLOW (BAR CHART) ===
    analysis_df["week"] = analysis_df["date"].dt.to_period("W").astype(str)
    weekly = analysis_df.groupby("week")[["tasks created", "tasks done"]].sum().reset_index()

    plt.figure(figsize=(14, 6))
    plt.bar(weekly["week"], -weekly["tasks created"], label="Tasks Created", color="red")
    plt.bar(weekly["week"], weekly["tasks done"], label="Tasks Done", color="green")
    plt.xlabel("Week")
    plt.ylabel("Number of Tasks")
    plt.title("Weekly Task Creation and Completion (All Time)")
    plt.xticks(rotation=90, fontsize=6)
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y')
    plt.savefig(os.path.join(output_folder, "task_flow_weekly.png"))
    plt.close()
    print("Saved: task_flow_weekly.png")

    # === WEEKLY EFFICIENCY RATIO (LINE CHART) ===
    def calc_efficiency(row):
        c, d = row["tasks created"], row["tasks done"]
        if c == 0 and d == 0:
            return 0
        elif d >= c:
            return d / c if c != 0 else float('inf')
        else:
            return -(c / d) if d != 0 else -float('inf')

    weekly["efficiency"] = weekly.apply(calc_efficiency, axis=1)

    # Bar colors based on sign
    colors = ["green" if val > 0 else "red" for val in weekly["efficiency"]]

    plt.figure(figsize=(14, 4))
    plt.bar(weekly["week"], weekly["efficiency"], color=colors, label="Efficiency Ratio")
    plt.axhline(0, color="black", linestyle="-", linewidth=1)
    plt.ylabel("Efficiency Ratio")
    plt.xlabel("Week")
    plt.title("Weekly Task Efficiency (Done vs. Created)")
    plt.xticks(rotation=90, fontsize=6)
    plt.grid(axis='y')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "task_efficiency_weekly.png"))
    plt.close()
    
        # === WEEKLY EFFICIENCY PER YEAR ===
    analysis_df["year"] = analysis_df["date"].dt.year
    analysis_df["week_number"] = analysis_df["date"].dt.isocalendar().week

    for year in sorted(analysis_df["year"].unique()):
        year_df = analysis_df[analysis_df["year"] == year]
        weekly_year = year_df.groupby("week_number")[["tasks created", "tasks done"]].sum().reset_index()

        # Fill missing weeks 1–52
        full_weeks = pd.DataFrame({"week_number": range(1, 53)})
        weekly_year = pd.merge(full_weeks, weekly_year, on="week_number", how="left").fillna(0)

        # Calculate efficiency
        def calc_eff(row):
            c, d = row["tasks created"], row["tasks done"]
            if c == 0 and d == 0:
                return 0
            elif d >= c:
                return d / c if c != 0 else float('inf')
            else:
                return -(c / d) if d != 0 else -float('inf')

        weekly_year["efficiency"] = weekly_year.apply(calc_eff, axis=1)
        colors = ["green" if val > 0 else "red" for val in weekly_year["efficiency"]]

        # Plot
        plt.figure(figsize=(14, 4))
        plt.bar(weekly_year["week_number"], weekly_year["efficiency"], color=colors)
        plt.axhline(0, color="black", linestyle="-", linewidth=1)
        plt.title(f"Weekly Task Efficiency – {year}")
        plt.xlabel("Week Number")
        plt.ylabel("Efficiency Ratio")
        plt.xticks(range(1, 53))
        plt.grid(axis='y')
        plt.tight_layout()

        filename = os.path.join(output_folder, f"task_efficiency_weekly_{year}.png")
        plt.savefig(filename)
        plt.close()
        print(f"Saved: {filename}")



    # === PER-YEAR WEEKLY TASK FLOW ===
    analysis_df["year"] = analysis_df["date"].dt.year
    analysis_df["week_number"] = analysis_df["date"].dt.isocalendar().week

    for year in sorted(analysis_df["year"].unique()):
        year_df = analysis_df[analysis_df["year"] == year]
        weekly_year = year_df.groupby("week_number")[["tasks created", "tasks done"]].sum().reset_index()

        # Fill in all 52 weeks
        full_weeks = pd.DataFrame({"week_number": range(1, 53)})
        weekly_year = pd.merge(full_weeks, weekly_year, on="week_number", how="left").fillna(0)

        plt.figure(figsize=(14, 6))
        plt.bar(weekly_year["week_number"], -weekly_year["tasks created"], label="Tasks Created", color="red")
        plt.bar(weekly_year["week_number"], weekly_year["tasks done"], label="Tasks Done", color="green")
        plt.xlabel("Week Number")
        plt.ylabel("Number of Tasks")
        plt.title(f"Weekly Task Creation and Completion – {year}")
        plt.xticks(range(1, 53))
        plt.legend()
        plt.tight_layout()
        plt.grid(axis='y')
        filename = os.path.join(output_folder, f"{year}_task_creation_weekly.png")
        plt.savefig(filename)
        plt.close()
        print(f"Saved: {filename}")


def plot_day_of_week_throughput(df, output_folder):

    done_df = df[df["Status"].str.lower() == "done"].copy()
    done_df["Last edited"] = pd.to_datetime(done_df["Last edited"], errors="coerce")
    done_df = done_df.dropna(subset=["Last edited"])

    done_df["weekday"] = done_df["Last edited"].dt.day_name()
    done_df["year"] = done_df["Last edited"].dt.year

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    def plot_weekday_bar(data, title, filename):
        counts = data["weekday"].value_counts().reindex(weekday_order, fill_value=0)
        percentages = (counts / counts.sum()) * 100
        plt.figure(figsize=(8, 4))
        plt.bar(percentages.index, percentages.values, color="skyblue")
        plt.title(title)
        plt.ylabel("Percentage of Tasks Done")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, filename))
        plt.close()
        print(f"Saved: {filename}")

    # All-time
    plot_weekday_bar(done_df, "Task Completion by Day of Week (All Time)", "throughput_dayofweek_all.png")

    # Per year
    for year in sorted(done_df["year"].unique()):
        year_df = done_df[done_df["year"] == year]
        plot_weekday_bar(year_df, f"Task Completion by Day of Week – {year}", f"throughput_dayofweek_{year}.png")

def plot_working_session_heatmaps(df, output_folder):
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    done_df = df[df["Status"].str.lower() == "done"].copy()
    done_df["Created time"] = pd.to_datetime(done_df["Created time"], errors="coerce")
    done_df["Last edited"] = pd.to_datetime(done_df["Last edited"], errors="coerce")
    done_df = done_df.dropna(subset=["Created time", "Last edited"])

    def generate_heatmap(time_col, label, filename_prefix):
        df_time = done_df.copy()
        df_time["weekday"] = df_time[time_col].dt.day_name()
        df_time["hour"] = df_time[time_col].dt.hour
        df_time["year"] = df_time[time_col].dt.year

        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        hour_bins = list(range(24))

        def plot_heat(data, year_label, filename):
            pivot = data.groupby(["weekday", "hour"]).size().unstack(fill_value=0).reindex(weekday_order)
            percent = pivot.div(pivot.sum().sum()) * 100
            plt.figure(figsize=(12, 5))
            sns.heatmap(percent, cmap="YlGnBu", annot=False, cbar_kws={"label": "% of Tasks"})
            plt.title(f"{label} Heatmap – {year_label}")
            plt.ylabel("Weekday")
            plt.xlabel("Hour of Day")
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, filename))
            plt.close()
            print(f"Saved: {filename}")

        # All time
        plot_heat(df_time, "All Time", f"{filename_prefix}_all.png")

        # Per year
        for year in sorted(df_time["year"].dropna().unique()):
            plot_heat(df_time[df_time["year"] == year], str(year), f"{filename_prefix}_{year}.png")

    # Creation heatmaps
    generate_heatmap("Created time", "Task Creation", "heatmap_creation")

    # Completion heatmaps
    generate_heatmap("Last edited", "Task Completion", "heatmap_completion")

def plot_abbvie_done_heatmaps(df, output_folder):
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # Filter for Abbvie & done tasks
    abbvie_df = df[
        df["Workspace"].str.contains("Abbvie", case=False, na=False) &
        (df["Status"].str.lower() == "done")
    ].copy()

    abbvie_df["Created time"] = pd.to_datetime(abbvie_df["Created time"], errors="coerce")
    abbvie_df["Last edited"] = pd.to_datetime(abbvie_df["Last edited"], errors="coerce")
    abbvie_df = abbvie_df.dropna(subset=["Created time", "Last edited"])

    # Limit to 2023–2025
    abbvie_df = abbvie_df[
        abbvie_df["Created time"].dt.year.isin([2023, 2024, 2025]) |
        abbvie_df["Last edited"].dt.year.isin([2023, 2024, 2025])
    ]

    def generate_heatmap(time_col, title, filename):
        df_time = abbvie_df.copy()
        df_time["weekday"] = df_time[time_col].dt.day_name()
        df_time["hour"] = df_time[time_col].dt.hour

        # Pivot to weekday × hour and normalize
        pivot = df_time.groupby(["weekday", "hour"]).size().unstack(fill_value=0)
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        pivot = pivot.reindex(weekday_order)
        percent = pivot.div(pivot.sum().sum()) * 100

        # Plot
        plt.figure(figsize=(12, 5))
        sns.heatmap(percent, cmap="YlGnBu", annot=False, cbar_kws={"label": "% of Tasks"})
        plt.title(f"Abbvie – {title} (Done Tasks, 2023–2025)")
        plt.ylabel("Weekday")
        plt.xlabel("Hour of Day")
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, filename))
        plt.close()
        print(f"Saved: {filename}")

    # Creation heatmap
    generate_heatmap("Created time", "Task Creation Heatmap", "abbvie_heatmap_creation.png")

    # Completion heatmap
    generate_heatmap("Last edited", "Task Completion Heatmap", "abbvie_heatmap_completion.png")

def plot_liberal_stuff_done_heatmaps(df, output_folder):
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # Filter for Liberal Stuff & done tasks
    liberal_df = df[
        df["Workspace"].str.contains("Liberal Stuff", case=False, na=False) &
        (df["Status"].str.lower() == "done")
    ].copy()

    liberal_df["Created time"] = pd.to_datetime(liberal_df["Created time"], errors="coerce")
    liberal_df["Last edited"] = pd.to_datetime(liberal_df["Last edited"], errors="coerce")
    liberal_df = liberal_df.dropna(subset=["Created time", "Last edited"])

    # Limit to 2023–2025
    liberal_df = liberal_df[
        liberal_df["Created time"].dt.year.isin([2023, 2024, 2025]) |
        liberal_df["Last edited"].dt.year.isin([2023, 2024, 2025])
    ]

    def generate_heatmap(time_col, title, filename):
        df_time = liberal_df.copy()
        df_time["weekday"] = df_time[time_col].dt.day_name()
        df_time["hour"] = df_time[time_col].dt.hour

        # Pivot to weekday × hour and normalize
        pivot = df_time.groupby(["weekday", "hour"]).size().unstack(fill_value=0)
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        pivot = pivot.reindex(weekday_order)
        percent = pivot.div(pivot.sum().sum()) * 100

        # Plot
        plt.figure(figsize=(12, 5))
        sns.heatmap(percent, cmap="YlGnBu", annot=False, cbar_kws={"label": "% of Tasks"})
        plt.title(f"Liberal Stuff – {title} (Done Tasks, 2023–2025)")
        plt.ylabel("Weekday")
        plt.xlabel("Hour of Day")
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, filename))
        plt.close()
        print(f"Saved: {filename}")

    # Creation heatmap
    generate_heatmap("Created time", "Task Creation Heatmap", "liberal_heatmap_creation.png")

    # Completion heatmap
    generate_heatmap("Last edited", "Task Completion Heatmap", "liberal_heatmap_completion.png")


# === USAGE EXAMPLE ===
folder_name = "04_Jun_2025"  # Change this to your actual folder path
csv_file = find_all_csv(folder_name)

if csv_file:
    df, analysis = analyze_tasks(csv_file)
    output_folder = os.path.dirname(csv_file)

    print("Loaded CSV Data:")
    print(df.head())
    print("\nTask Analysis:")
    print(analysis)

    plot_time_to_completion_histogram(df, output_path=os.path.join(output_folder, "TTC_All_time.png"))
    plot_monthly_task_flow(analysis, output_path=os.path.join(output_folder, "Task_flow.png"))
    waterfall(df, analysis, output_prefix=os.path.join(output_folder, "waterfall"))
    workspace_piecharts_by_year(df, output_folder)
    plot_median_ttc_over_time(df, output_folder)
    plot_task_flow_time_series(analysis, output_folder)
    plot_day_of_week_throughput(df, output_folder)
    plot_working_session_heatmaps(df, output_folder)
    plot_abbvie_done_heatmaps(df, output_folder)
    plot_liberal_stuff_done_heatmaps(df, output_folder)
