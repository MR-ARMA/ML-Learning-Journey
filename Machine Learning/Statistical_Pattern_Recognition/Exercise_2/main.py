# ===============================================================
# Experiment 2.5: Generate X4 and visualize + classify
# ===============================================================

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import multivariate_normal

# -------------------------
# 1. Create 'results' folder
# -------------------------
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# -------------------------
# 2. Define parameters
# -------------------------
N = 1000  # total samples
K = 3     # number of classes
N_per_class = N // K

means = np.array([
    [1, 1],
    [10, 5],
    [11, 1]
])

cov = np.array([
    [7, 4],
    [4, 5]
])

# -------------------------
# 3. Generate data
# -------------------------
X = np.zeros((N, 2))
labels = np.zeros(N, dtype=int)

for i in range(K):
    start = i * N_per_class
    end = (i + 1) * N_per_class
    X[start:end, :] = np.random.multivariate_normal(means[i], cov, N_per_class)
    labels[start:end] = i

# -------------------------
# 4. Save dataset
# -------------------------
np.savez(results_dir / "X4_dataset.npz", X=X, labels=labels)
print(f"✅ Dataset saved to {results_dir / 'X4_dataset.npz'}")

# -------------------------
# 5. Visualization
# -------------------------
colors = ['r', 'g', 'b']
plt.figure(figsize=(8, 6))
for i in range(K):
    plt.scatter(X[labels == i, 0], X[labels == i, 1],
                c=colors[i], s=20, label=f'Class {i+1}', alpha=0.6)

plt.title("Generated Data Set X₄")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(results_dir / "X4_scatter.png", dpi=300)
plt.show()
print(f"✅ Plot saved to {results_dir / 'X4_scatter.png'}")

# -------------------------
# 6. (Optional) Bayesian classification and decision boundaries
# -------------------------
x1, x2 = np.meshgrid(np.linspace(-5, 20, 300), np.linspace(-5, 15, 300))
pos = np.dstack((x1, x2))

pdfs = [multivariate_normal(mean=means[i], cov=cov).pdf(pos) for i in range(K)]
posteriors = np.array(pdfs)
decision_map = np.argmax(posteriors, axis=0)

plt.figure(figsize=(8, 6))
plt.contourf(x1, x2, decision_map, alpha=0.3, levels=K-1, cmap='rainbow')
for i in range(K):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], c=colors[i], label=f'Class {i+1}', s=15)
plt.title("Decision Regions for X₄")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(results_dir / "X4_decision_regions.png", dpi=300)
plt.show()
print(f"✅ Decision region plot saved to {results_dir / 'X4_decision_regions.png'}")
