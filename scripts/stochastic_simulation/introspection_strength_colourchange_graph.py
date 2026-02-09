import pyarrow.parquet as pq
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pq.read_table("numerical_introspection_stationary_distribution.parquet").to_pandas()

x = df["introspection_strength"]

# --- Colours ---
BG_COLOUR = "#02134A"
FG_COLOUR = "#D5FAFF"
FLURO_COLOURS = [
    "#00e5ff",  # cyan
    "#ff00ff",  # magenta
    "#39ff14",  # neon green
    "#ffea00",  # neon yellow
    "#ff6f00",  # neon orange
    "#ff1744"   # neon red
]

# --- Figure setup ---
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(BG_COLOUR)
ax.set_facecolor(BG_COLOUR)

# --- Plot lines ---
for i, col in enumerate(df.columns):
    if col == "introspection_strength":
        continue
    ax.plot(
        x,
        df[col],
        label=col,
        color=FLURO_COLOURS[i % len(FLURO_COLOURS)],
        linewidth=2.2
    )

# --- Labels and title ---
ax.set_xlabel("Introspection strength, δ", color=FG_COLOUR, fontsize=12)
ax.set_ylabel("Probability", color=FG_COLOUR, fontsize=12)
ax.set_title("Stationary distribution of strategy-pair probabilities",
             color=FG_COLOUR, fontsize=14, pad=10)


# --- Ticks ---
ax.tick_params(colors=FG_COLOUR)

# --- Spines ---
for spine in ax.spines.values():
    spine.set_color(FG_COLOUR)

# --- Legend ---
legend = ax.legend(ncol=2, fontsize=8, frameon=True)
legend.get_frame().set_facecolor(BG_COLOUR)
legend.get_frame().set_edgecolor(FG_COLOUR)
for text in legend.get_texts():
    text.set_color(FG_COLOUR)

plt.tight_layout()
plt.show()