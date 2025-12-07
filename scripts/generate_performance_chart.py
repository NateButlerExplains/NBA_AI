"""
generate_performance_chart.py

Generates predictor performance visualization charts for documentation.

Usage:
    python scripts/generate_performance_chart.py --season=2024-2025 --output=predictor_performance.png
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import log_loss, mean_absolute_error

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import config

# Configuration
DB_PATH = config["database"]["path"]


def fetch_predictor_data(seasons, predictors):
    """
    Fetches data from the database for the specified seasons and predictors.

    Args:
        seasons (list): List of seasons (e.g., ['2023-2024']).
        predictors (list): List of predictors (e.g., ['Baseline', 'Linear']).

    Returns:
        DataFrame: Data containing game_id, predictor, prediction_set, home_score, away_score.
    """
    with sqlite3.connect(DB_PATH) as conn:
        query = f"""
        SELECT p.game_id, p.predictor, p.prediction_set, gs.home_score, gs.away_score
        FROM Predictions p
        JOIN Games g ON p.game_id = g.game_id
        JOIN GameStates gs ON p.game_id = gs.game_id AND gs.is_final_state = True
        WHERE g.season IN ({','.join('?' for _ in seasons)})
          AND p.predictor IN ({','.join('?' for _ in predictors)});
        """
        df = pd.read_sql_query(query, conn, params=seasons + predictors)

    # Expand the JSON column 'prediction_set'
    df["prediction_set"] = df["prediction_set"].apply(json.loads)
    expanded_df = df.join(pd.json_normalize(df["prediction_set"]))
    expanded_df = expanded_df.drop(columns=["prediction_set"])

    # Calculate actual_home_margin
    expanded_df["actual_home_margin"] = (
        expanded_df["home_score"] - expanded_df["away_score"]
    )

    # Use current prediction schema column names (pred_* not predicted_*)
    expanded_df["pred_home_margin"] = (
        expanded_df["pred_home_score"] - expanded_df["pred_away_score"]
    )

    # Calculate actual_home_win_pct
    expanded_df["actual_home_win_pct"] = expanded_df.apply(
        lambda row: (
            1
            if row["home_score"] > row["away_score"]
            else (0 if row["home_score"] < row["away_score"] else 0.5)
        ),
        axis=1,
    )

    return expanded_df


def calculate_metrics(data):
    """
    Calculate Log Loss and MAE metrics for the given data.

    Args:
        data (DataFrame): Data containing actual and predicted values.

    Returns:
        dict: Dictionary of metrics for each predictor.
    """
    metrics = {}
    for predictor in data["predictor"].unique():
        df_pred = data[data["predictor"] == predictor]

        # Log Loss for home_win_pct
        log_loss_home_win_pct = log_loss(
            df_pred["actual_home_win_pct"], df_pred["pred_home_win_pct"]
        )

        # MAE for home_score, away_score, and margin (using pred_* column names)
        mae_home_score = mean_absolute_error(
            df_pred["home_score"], df_pred["pred_home_score"]
        )
        mae_away_score = mean_absolute_error(
            df_pred["away_score"], df_pred["pred_away_score"]
        )
        mae_home_margin = mean_absolute_error(
            df_pred["actual_home_margin"], df_pred["pred_home_margin"]
        )

        metrics[predictor] = {
            "Home Win Prob Log Loss": log_loss_home_win_pct,
            "Home Score MAE": mae_home_score,
            "Away Score MAE": mae_away_score,
            "Home Margin MAE": mae_home_margin,
        }
    return metrics


def plot_metrics(metrics, save=False, image_filename=None):
    """
    Generate dual-panel bar chart visualization of predictor performance.

    Args:
        metrics (dict): Dictionary of metrics by predictor.
        save (bool): Whether to save the plot to file.
        image_filename (str): Filename for saved plot (without extension).
    """
    # Convert the metrics dictionary into a DataFrame and round values
    df = pd.DataFrame(metrics).T.round(2)
    df_unstacked = df.unstack().reset_index()
    df_unstacked.columns = ["Metric", "Predictor", "Value"]

    # Apply custom labels for the metrics
    custom_labels = {
        "Home Score MAE": "Home Score\nMAE",
        "Away Score MAE": "Away Score\nMAE",
        "Home Margin MAE": "Home Margin\nMAE",
        "Home Win Prob Log Loss": "Home Win %\nLog Loss",
    }
    df_unstacked["Metric"] = df_unstacked["Metric"].map(custom_labels)

    # Split DataFrame into MAE and Log Loss data
    mae_df = df_unstacked[df_unstacked["Metric"].str.contains("MAE")]
    log_loss_df = df_unstacked[df_unstacked["Metric"].str.contains("Log Loss")]

    # Generate custom color palette
    deep_palette = sns.color_palette("deep")
    custom_palette = {
        key: deep_palette[i] for i, key in enumerate(df_unstacked["Predictor"].unique())
    }
    custom_palette["MLP"] = deep_palette[0]
    custom_palette["Tree"] = deep_palette[2]
    custom_palette["Linear"] = deep_palette[1]
    custom_palette["Baseline"] = deep_palette[7]

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [3, 1]}
    )

    # Plot MAE metrics
    sns.barplot(
        data=mae_df,
        x="Metric",
        y="Value",
        hue="Predictor",
        ax=ax1,
        palette=custom_palette,
    )
    ax1.set_ylabel("MAE (Points)", fontsize=18, fontweight="bold")
    ax1.tick_params(axis="y", labelsize=16)
    ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=18, fontweight="bold")
    ax1.grid(True)
    ax1.set_xlabel("")
    ax1.set_title("")
    ax1.xaxis.grid(False)

    # Annotate values on MAE bars
    for p in ax1.patches:
        value = p.get_height()
        if value > 0:
            ax1.text(
                p.get_x() + p.get_width() / 2,
                value - 0.8,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=14,
                weight="bold",
                color="white",
            )

    # Plot Log Loss metrics
    sns.barplot(
        data=log_loss_df,
        x="Metric",
        y="Value",
        hue="Predictor",
        ax=ax2,
        palette=custom_palette,
    )
    ax2.set_ylabel(
        "Log Loss", rotation=270, labelpad=24, fontsize=18, fontweight="bold"
    )
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.tick_params(axis="y", labelsize=16)
    ax2.set_xticklabels(log_loss_df["Metric"], fontsize=18, fontweight="bold")
    ax2.grid(True)
    ax2.set_xlabel("")
    ax2.set_title("")
    ax2.xaxis.grid(False)

    # Annotate values on Log Loss bars
    for p in ax2.patches:
        value = p.get_height()
        if value > 0:
            ax2.text(
                p.get_x() + p.get_width() / 2,
                value - 0.04,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=14,
                weight="bold",
                color="white",
            )

    # Add a super title for the figure
    fig.suptitle("Prediction Engine Performance", fontsize=24, weight="bold", y=1)

    # Customize and add legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(df_unstacked["Predictor"].unique()),
        title="Predictors",
        bbox_to_anchor=(0.5, 0.95),
        fontsize=16,
        title_fontproperties={"weight": "bold", "size": 18},
    )
    ax1.get_legend().remove()
    ax2.get_legend().remove()

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    # Save the plot to a file if requested
    if save and image_filename:
        plt.savefig(f"{image_filename}.png", dpi=300, bbox_inches="tight")
        print(f"✅ Chart saved to {image_filename}.png")

    # Display the plot
    plt.show()


def main():
    """Main entry point for generating performance charts."""
    parser = argparse.ArgumentParser(
        description="Generate predictor performance visualization charts"
    )
    parser.add_argument(
        "--season",
        type=str,
        default="2024-2025",
        help="Season to analyze (default: 2024-2025)",
    )
    parser.add_argument(
        "--predictors",
        type=str,
        nargs="+",
        default=["Baseline", "Linear", "Tree", "MLP"],
        help="Predictors to include (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictor_performance",
        help="Output filename (without extension, default: predictor_performance)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Display only, don't save to file",
    )

    args = parser.parse_args()

    print(f"Fetching data for {args.season} season...")
    data = fetch_predictor_data([args.season], args.predictors)

    if data.empty:
        print(
            f"❌ No data found for season {args.season} with predictors {args.predictors}"
        )
        return

    print(f"Calculating metrics for {len(data)} games...")
    metrics = calculate_metrics(data)

    # Display metrics
    print("\n" + "=" * 60)
    print("PREDICTOR PERFORMANCE METRICS")
    print("=" * 60)
    for predictor, metric_dict in metrics.items():
        print(f"\n{predictor}:")
        for metric, value in metric_dict.items():
            print(f"  {metric}: {value:.2f}")
    print("=" * 60 + "\n")

    # Generate plot
    print("Generating visualization...")
    plot_metrics(metrics, save=not args.no_save, image_filename=args.output)


if __name__ == "__main__":
    main()
