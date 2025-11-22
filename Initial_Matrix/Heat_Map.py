import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------
# 1) Load transition matrix
# -------------------

base_dir = os.path.dirname(os.path.abspath(__file__))
matrix_path = os.path.join(base_dir, "jpm_transition_matrix.csv")
df = pd.read_csv(matrix_path, index_col=0)

matrix = df.to_numpy()
labels = df.columns.to_list()

# -------------------
# 2) Create heatmap
# -------------------

plt.figure(figsize=(14, 12))

plt.imshow(
    matrix,
    cmap="viridis",
    aspect='equal',
    interpolation='none'
)

plt.title("JPM Transition Probability Matrix by % Return Bin", fontsize=14, weight="bold", pad=15)
plt.xlabel("Next Day Return Interval (%)", fontsize=12)
plt.ylabel("Today Return Interval (%)", fontsize=12)


plt.title("JPM Transition Probability Matrix by % Return Bin", fontsize=14, weight="bold", pad=15)
plt.xlabel("Next Day Return Interval (%)", fontsize=12)
plt.ylabel("Today Return Interval (%)", fontsize=12)

# Ticks
step = max(1, len(labels)//20)  # limit number of tick labels for clarity
plt.xticks(np.arange(0, len(labels), step), labels[::step], rotation=90, fontsize=7)
plt.yticks(np.arange(0, len(labels), step), labels[::step], fontsize=7)

# Colorbar
cbar = plt.colorbar()
cbar.set_label("Transition Probability", rotation=270, labelpad=15)

plt.tight_layout()

# -------------------
# 3) Save & show
# -------------------
output_path = os.path.join(base_dir, "jpm_transition_heatmap.png")
plt.savefig(output_path, dpi=300)
plt.show()

print(f"✅ Heatmap saved to: {output_path}")
