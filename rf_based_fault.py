"""
BLDC Fault Detection — Final RF Pipeline
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

FILE = "BLDC_4Fault_Dataset.xlsx"
FEATURES = ["Voltage", "Current", "Temperature", "Vibration", "Speed", "Torque"]
LABEL_COL = "Condition"

FAULT_ORDER = [
    "Normal",
    "Bearing Fault",
    "Single Winding Fault",
    "Double Winding Fault"
]

COLORS = ["green", "orange", "blue", "red"]
NOISE_SCALE = 0.7
RANDOM_SEED = 42

legend_elements = [
    mpatches.Patch(facecolor=c, label=l)
    for c, l in zip(COLORS, FAULT_ORDER)
]

df_raw = pd.read_excel(FILE)

print("=" * 60)
print("  BLDC Fault Detection — Final RF Pipeline")
print("=" * 60)
print(f"\nDataset : {len(df_raw)} rows × {len(df_raw.columns)} cols")
print(f"Classes : {df_raw[LABEL_COL].value_counts().to_dict()}")

rng = np.random.default_rng(RANDOM_SEED)
df = df_raw.copy()

print(f"\nAdding Gaussian noise  (scale = {NOISE_SCALE} × feature std)")

for feat in FEATURES:
    sigma = df[feat].std() * NOISE_SCALE
    df[feat] = df[feat] + rng.normal(0, sigma, size=len(df))

le = LabelEncoder()
le.fit(FAULT_ORDER)

X = df[FEATURES].values
y = le.transform(df[LABEL_COL])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1
    ))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

cv_results = cross_validate(
    pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring=["accuracy", "f1_macro"],
    return_train_score=True
)

cv_val_acc = cv_results["test_accuracy"]
cv_train_acc = cv_results["train_accuracy"]
cv_f1 = cv_results["test_f1_macro"]

print("\n5-Fold Cross-Validation Results")
for i, acc in enumerate(cv_val_acc, 1):
    print(f"Fold {i} Accuracy: {acc * 100:.2f}%")

print(f"CV Mean Accuracy: {cv_val_acc.mean() * 100:.2f}%")

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

test_acc = accuracy_score(y_test, y_pred) * 100

print("\nFinal Test Accuracy:", round(test_acc, 2), "%")
print(classification_report(y_test, y_pred, target_names=FAULT_ORDER, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(7, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=FAULT_ORDER)
disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=20)

ax.set_title(
    f"RF-Based Confusion Matrix\nTest Accuracy: {test_acc:.2f}%",
    fontsize=13
)

plt.tight_layout()
plt.savefig("RF_Confusion_Matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# Feature Importance
importances = pipeline.named_steps["clf"].feature_importances_
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 5))
bars = plt.bar(
    [FEATURES[i] for i in sorted_idx],
    importances[sorted_idx],
    color="steelblue"
)

plt.title("RF-Based Feature Importance", fontsize=14)
plt.xlabel("Feature", fontsize=12)
plt.ylabel("Importance", fontsize=12)
plt.xticks(rotation=15, fontsize=10, fontweight="bold")
plt.yticks(fontsize=10, fontweight="bold")

for bar in bars:
    h = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        h + 0.003,
        f"{h:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold"
    )

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("RF_Feature_Importance.png", dpi=300, bbox_inches="tight")
plt.show()

# Test output dataframe
df_test = pd.DataFrame(X_test, columns=FEATURES)
df_test["RF_Prediction"] = le.inverse_transform(y_pred)
df_test["Condition"] = le.inverse_transform(y_test)

df_test["Condition"] = pd.Categorical(
    df_test["Condition"],
    categories=FAULT_ORDER,
    ordered=True
)

df_test["RF_Prediction"] = pd.Categorical(
    df_test["RF_Prediction"],
    categories=FAULT_ORDER,
    ordered=True
)

# Feature Graphs
for feat in FEATURES:

    plt.figure(figsize=(12, 6))

    avg_values = (
        df_test.groupby("RF_Prediction", observed=False)[feat]
        .mean()
        .reindex(FAULT_ORDER)
    )

    bars = plt.bar(avg_values.index, avg_values.values, color=COLORS)

    plt.title(f"RF-Based: {feat} under Different Conditions", fontsize=14)
    plt.xlabel("Predicted Fault Type", fontsize=12)
    plt.ylabel(feat, fontsize=12)

    plt.xticks(rotation=15, fontsize=10, fontweight="bold")
    plt.yticks(fontsize=10, fontweight="bold")

    for bar in bars:
        h = bar.get_height()
        if not np.isnan(h):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold"
            )

    legend = plt.legend(
        handles=legend_elements,
        title="Fault Types",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        prop={"weight": "bold", "size": 10}
    )

    legend.get_title().set_fontweight("bold")

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.grid(False)
    plt.ylim(0, avg_values.max() * 1.25)
    plt.tight_layout(rect=[0, 0, 0.82, 1])

    plt.savefig(f"RF_{feat}_Graph.png", dpi=300, bbox_inches="tight")
    plt.show()

# Accuracy Graph
plt.figure(figsize=(8, 5))

x_labels = [f"Fold {i}" for i in range(1, 6)] + ["CV Mean", "Test"]
y_vals = list(cv_val_acc * 100) + [cv_val_acc.mean() * 100, test_acc]

bars = plt.bar(x_labels, y_vals, width=0.5)

plt.title("RF-Based Accuracy by Fold & Final Test Set", fontsize=13)
plt.xlabel("Split", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.ylim(0, 115)

for bar in bars:
    h = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        h + 0.5,
        f"{h:.2f}%",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold"
    )

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.xticks(rotation=15, fontsize=9, fontweight="bold")
plt.tight_layout()
plt.savefig("RF_Accuracy_Graph.png", dpi=300, bbox_inches="tight")
plt.show()

df_out = df_test.sort_values(["Condition", "RF_Prediction"]).reset_index(drop=True)
df_out.to_excel("RF_Based_Output.xlsx", index=False)

print("\nAll RF-based graphs and output files generated successfully.")