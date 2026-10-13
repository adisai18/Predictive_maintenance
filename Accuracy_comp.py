"""
BLDC Fault Detection — Model Comparison Chart
==============================================
Compares final test accuracy across all trained models.
Edit MODEL_NAMES / MODEL_ACCURACIES below if your labels or numbers change.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.weight"]      = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

# ── EDIT HERE ────────────────────────────────────────────────────────────
MODEL_NAMES      = ["conventional", "ANN", "Random Forest"]
MODEL_ACCURACIES = [78.9, 91.0, 93.5]
MODEL_COLORS     = ["#E07B39", "mediumseagreen", "#3A7FBF"]
# ─────────────────────────────────────────────────────────────────────────

plt.figure(figsize=(8, 6))
bars = plt.bar(MODEL_NAMES, MODEL_ACCURACIES, color=MODEL_COLORS, width=0.5)

plt.title("Model Comparison — Test Accuracy", fontsize=15)
plt.xlabel("Model", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.ylim(0, 110)

for bar, acc in zip(bars, MODEL_ACCURACIES):
    h = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        h + 1.5,
        f"{acc:.2f}%",
        ha="center", va="bottom",
        fontsize=12, fontweight="bold"
    )

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.xticks(fontsize=11, fontweight="bold")
plt.yticks(fontsize=10, fontweight="bold")
plt.tight_layout()

plt.savefig("Model_Comparison_Accuracy.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved: Model_Comparison_Accuracy.png")