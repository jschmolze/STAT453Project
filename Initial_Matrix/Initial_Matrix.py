import numpy as np
import pandas as pd
import os

# -------------------
# 1) Load JPM %Return data from CSV
# -------------------

base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "..", "data", "PriceHistoryJPM.csv")

df = pd.read_csv(data_path)

possible_return_cols = [col for col in df.columns if "%return" in col.lower() or "return" in col.lower()]
if not possible_return_cols:
    raise ValueError("No '%Return' column found in PriceHistoryJPM.csv.")
return_col = possible_return_cols[0]
returns = df[return_col].dropna().to_numpy(dtype=float)

# -------------------
# 2) Construct safe monotonic edges (explicit and ordered)
# -------------------

# central region
central_left  = -1.05
central_right =  1.05

edges = np.concatenate([

    np.array([-np.inf]),

    np.arange(-10.0, -5.0, 1.0),

    np.arange(-5.0, -2.05, 0.5),

    # exact match at seams
    np.linspace(-2.05, central_left, 6),

    np.linspace(central_left, central_right, 22),

    np.linspace(central_right, 2.05, 6),

    np.arange(2.05, 5.01, 0.5),

    np.arange(5.0, 10.01, 1.0),

    np.array([np.inf])
])


edges = np.unique(edges)
edges.sort()

# SAFETY CHECK — REQUIRED
widths = edges[1:] - edges[:-1]
if not np.all(widths > 0):
    raise RuntimeError("ERROR: zero-width bins detected in edge construction.")

n_bins = len(edges) - 1

# -------------------
# 3) Digitize to bins
# -------------------

bins_today = np.digitize(returns[:-1], edges) - 1
bins_next  = np.digitize(returns[1:], edges) - 1

# -------------------
# 4) Transition matrix
# -------------------

transition_counts = np.zeros((n_bins, n_bins), dtype=float)

for i, j in zip(bins_today, bins_next):
    if 0 <= i < n_bins and 0 <= j < n_bins:
        transition_counts[i, j] += 1

row_sums = transition_counts.sum(axis=1, keepdims=True)

# Normalize *without artificially inflating empty rows*
transition_probs = np.zeros_like(transition_counts)
nonzero = (row_sums[:,0] > 0)
transition_probs[nonzero] = transition_counts[nonzero] / row_sums[nonzero]

# -------------------
# 5) Bin labels
# -------------------

bin_labels = []
for i in range(n_bins):
    low, high = edges[i], edges[i+1]
    if np.isinf(low):
        label = f"< {high:.2f}%"
    elif np.isinf(high):
        label = f"> {low:.2f}%"
    else:
        label = f"[{low:.2f}%, {high:.2f}%)"
    bin_labels.append(label)

# -------------------
# 6) Save outputs
# -------------------

pd.DataFrame({"Percent Change (%)": returns}).to_csv(
    os.path.join(base_dir, "jpm_percent_change.csv"), index_label="Index"
)

pd.DataFrame(
    transition_probs,
    index=bin_labels,
    columns=bin_labels
).to_csv(os.path.join(base_dir, "jpm_transition_matrix.csv"))

print(f"Transition matrix complete with {n_bins} bins.")
print("Saved:")
print(f"- {os.path.join(base_dir, 'jpm_percent_change.csv')}")
print(f"- {os.path.join(base_dir, 'jpm_transition_matrix.csv')}")
