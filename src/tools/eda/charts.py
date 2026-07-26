"""Chart generation for EDA tool using matplotlib.

Generates transaction analysis visualizations saved as PNG files.
Uses Agg backend for headless rendering (compatible with server-side
execution and future Streamlit integration).

All charts use a consistent dark theme with high DPI for clarity.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pandas import DataFrame

from loguru import logger

# Consistent chart styling
CHART_STYLE = {
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e94560",
    "axes.labelcolor": "#eee",
    "text.color": "#eee",
    "xtick.color": "#aaa",
    "ytick.color": "#aaa",
    "grid.color": "#333",
    "grid.alpha": 0.3,
}
CHART_DPI = 150
ACCENT_COLOR = "#e94560"
SECONDARY_COLOR = "#0f3460"
TERTIARY_COLOR = "#533483"
NORMAL_COLOR = "#0f3460"
FRAUD_COLOR = "#e94560"


def _style_figure(fig: plt.Figure, axes: list[plt.Axes] | plt.Axes) -> None:
    """Apply dark theme styling directly to figure and axes objects (object-oriented Matplotlib)."""
    fig.patch.set_facecolor("#1a1a2e")
    ax_list = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    for ax in ax_list:
        ax.set_facecolor("#16213e")
        ax.spines['bottom'].set_color('#e94560')
        ax.spines['top'].set_color('#e94560')
        ax.spines['left'].set_color('#e94560')
        ax.spines['right'].set_color('#e94560')
        ax.xaxis.label.set_color('#eee')
        ax.yaxis.label.set_color('#eee')
        ax.title.set_color('#eee')
        ax.tick_params(colors='#aaa')


def plot_transaction_volume_timeline(
    df: DataFrame,
    output_path: Path,
    timestamp_col: str = "timestamp",
) -> str:
    """Bar chart of transaction count per timestamp period.

    The AMLSim dataset uses integer day-indices (0-199) as timestamps.
    Groups transactions by timestamp and plots volume over time.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    _style_figure(fig, ax)

    volume = df.groupby(timestamp_col).size()
    ax.bar(volume.index, volume.values, color=ACCENT_COLOR, alpha=0.85, width=0.8)
    ax.set_xlabel("Time Period (day index)")
    ax.set_ylabel("Transaction Count")
    ax.set_title("Transaction Volume Over Time")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)

    logger.debug("Chart saved: {path}", path=str(output_path))
    return str(output_path)


def plot_amount_distribution(
    df: DataFrame,
    output_path: Path,
    amount_col: str = "tx_amount",
) -> str:
    """Histogram of transaction amounts on a log scale.

    Uses logarithmic binning to handle the wide range of amounts
    in AML data (from near-zero to millions).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _style_figure(fig, axes)

    amounts = df[amount_col].dropna()
    positive_amounts = amounts[amounts > 0]

    # Linear scale histogram
    ax1 = axes[0]
    ax1.hist(amounts.values, bins=50, color=ACCENT_COLOR, alpha=0.85, edgecolor="#1a1a2e")
    ax1.set_xlabel("Transaction Amount")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Amount Distribution (Linear)")
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Log scale histogram for better visibility of distribution tail
    ax2 = axes[1]
    if len(positive_amounts) > 0:
        log_bins = np.logspace(
            np.log10(max(positive_amounts.min(), 1)),
            np.log10(positive_amounts.max()),
            50,
        )
        ax2.hist(positive_amounts.values, bins=log_bins, color=TERTIARY_COLOR, alpha=0.85, edgecolor="#1a1a2e")
        ax2.set_xscale("log")
    ax2.set_xlabel("Transaction Amount (log scale)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Amount Distribution (Log Scale)")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)

    logger.debug("Chart saved: {path}", path=str(output_path))
    return str(output_path)


def plot_fraud_vs_normal_comparison(
    df: DataFrame,
    output_path: Path,
    amount_col: str = "tx_amount",
    fraud_col: str = "is_fraud",
) -> str:
    """Side-by-side comparison of fraud vs normal transaction amount distributions.

    Overlays histograms for fraud and legitimate transactions to highlight
    differences in their amount distributions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _style_figure(fig, axes)

    normal = df[df[fraud_col] == False][amount_col].dropna()
    fraud = df[df[fraud_col] == True][amount_col].dropna()

    # Amount distribution comparison
    ax1 = axes[0]
    if len(normal) > 0:
        ax1.hist(normal.values, bins=50, color=NORMAL_COLOR, alpha=0.7, label="Normal", edgecolor="#1a1a2e")
    if len(fraud) > 0:
        ax1.hist(fraud.values, bins=50, color=FRAUD_COLOR, alpha=0.7, label="Fraud", edgecolor="#1a1a2e")
    ax1.set_xlabel("Transaction Amount")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Amount Distribution: Fraud vs Normal")
    ax1.legend()

    # Transaction count comparison (simple bar chart)
    ax2 = axes[1]
    categories = ["Normal", "Fraud"]
    counts = [len(normal), len(fraud)]
    bars = ax2.bar(categories, counts, color=[NORMAL_COLOR, FRAUD_COLOR], alpha=0.85, edgecolor="#1a1a2e")
    ax2.set_ylabel("Transaction Count")
    ax2.set_title("Transaction Count: Fraud vs Normal")
    for bar, count in zip(bars, counts):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
            f"{count:,}", ha="center", va="bottom", fontsize=10, color="#eee",
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)

    logger.debug("Chart saved: {path}", path=str(output_path))
    return str(output_path)


def plot_top_senders(
    df: DataFrame,
    output_path: Path,
    top_n: int = 20,
) -> str:
    """Horizontal bar chart of accounts with highest outgoing transaction volume.

    Useful for identifying high-activity senders that may warrant further analysis.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    _style_figure(fig, ax)

    sender_counts = df["sender_account_id"].value_counts().head(top_n)
    y_positions = range(len(sender_counts))
    ax.barh(y_positions, sender_counts.values, color=ACCENT_COLOR, alpha=0.85)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"Acct {aid}" for aid in sender_counts.index], fontsize=8)
    ax.set_xlabel("Transaction Count")
    ax.set_title(f"Top {top_n} Senders by Transaction Count")
    ax.invert_yaxis()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)

    logger.debug("Chart saved: {path}", path=str(output_path))
    return str(output_path)
