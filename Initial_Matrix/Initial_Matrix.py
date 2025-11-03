import numpy as np
import pandas as pd
import yfinance as yf

# -------------------
# 1) Download JPM data
# -------------------
data = yf.download("JPM", start="2010-01-01", progress=False, auto_adjust=False)
returns = np.ravel(data["Close"].pct_change().mul(100).dropna().to_numpy())

# -------------------
# 2) Define basket edges
# -------------------

edges = []

# --- Central region [-1.05, 1.05] (21 baskets evenly spaced) ---
edges += list(np.linspace(-1.05, 1.05, 22))

# --- [-2.05, -1.06] and [1.06, 2.05] (5 baskets each) ---
neg_small = np.linspace(-2.05, -1.06, 6)
pos_small = np.linspace(1.06, 2.05, 6)
edges = sorted(set(edges + list(neg_small) + list(pos_small)))

# --- (-5, -2.05] and [2.05, 5) (0.5% width) ---
neg_mid = np.arange(-5.0, -2.05, 0.5).tolist()
pos_mid = np.arange(2.05, 5.01, 0.5).tolist()
edges = sorted(set(edges + neg_mid + pos_mid))

# --- (-10, -5.01] and [5.01, 10) (1% width) ---
neg_large = np.arange(-10.0, -5.0, 1.0).tolist()
pos_large = np.arange(5.0, 10.01, 1.0).tolist()
edges = sorted(set(edges + neg_large + pos_large))

# --- Tails: below -10 and above 10 ---
edges = [-np.inf] + edges + [np.inf]

edges = np.array(sorted(edges))
n_bins = len(edges) - 1

# -------------------
# 3) Bin each daily return
# -------------------
bins_today = np.digitize(returns[:-1], edges) - 1
bins_next = np.digitize(returns[1:], edges) - 1

# -------------------
# 4) Build transition frequency matrix
# -------------------
transition_counts = np.zeros((n_bins, n_bins), dtype=np.float64)
for i, j in zip(bins_today, bins_next):
    if 0 <= i < n_bins and 0 <= j < n_bins:
        transition_counts[i, j] += 1

# -------------------
# 5) Normalize to get probabilities
# -------------------
row_sums = transition_counts.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
transition_probs = transition_counts / row_sums

# -------------------
# 6) Save outputs
# -------------------
pd.DataFrame({"Percent Change (%)": returns}).to_csv("jpm_percent_change.csv", index_label="Index")
pd.DataFrame(
    transition_probs,
    columns=[f"To_{i}" for i in range(n_bins)],
    index=[f"From_{i}" for i in range(n_bins)]
).to_csv("jpm_transition_matrix.csv")

print(f"✅ Transition matrix complete with {n_bins} bins.")
print("Files saved as:")
print("  jpm_percent_change.csv")
print("  jpm_transition_matrix.csv")
