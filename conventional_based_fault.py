import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from matplotlib.patches import Patch

# ADD THESE 3 LINES HERE
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

df = pd.read_excel("BLDC_4Fault_Dataset.xlsx")

features = ["Voltage", "Current", "Temperature", "Vibration", "Speed", "Torque"]

fault_order = [
    "Normal",
    "Bearing Fault",
    "Single Winding Fault",
    "Double Winding Fault"
]

colors = ["green", "orange", "blue", "red"]

legend_elements = [
    Patch(facecolor="green", label="Normal"),
    Patch(facecolor="orange", label="Bearing Fault"),
    Patch(facecolor="blue", label="Single Winding Fault"),
    Patch(facecolor="red", label="Double Winding Fault")
]


def predict_fault(voltage, current, temperature, vibration, speed, torque):

    if (46 <= voltage <= 50 and
        6 <= current <= 9 and
        35 <= temperature <= 50 and
        0.1 <= vibration <= 0.5 and
        2920 <= speed <= 3080 and
        3.9 <= torque <= 4.5):
        return "Normal"

    elif (45 <= voltage <= 48 and
          9 <= current <= 12 and
          50 <= temperature <= 65 and
          1.5 <= vibration <= 2.5 and
          2620 <= speed <= 2880 and
          3.1 <= torque <= 3.8):
        return "Bearing Fault"

    elif (40 <= voltage <= 44 and
          13 <= current <= 16 and
          70 <= temperature <= 85 and
          0.8 <= vibration <= 1.5 and
          1850 <= speed <= 2350 and
          2.1 <= torque <= 3.0):
        return "Single Winding Fault"

    elif (34 <= voltage <= 40 and
          17 <= current <= 22 and
          88 <= temperature <= 110 and
          2.5 <= vibration <= 4.0 and
          850 <= speed <= 1450 and
          1.1 <= torque <= 2.0):
        return "Double Winding Fault"

    else:
        return "Unknown"


df["conventional_Prediction"] = df.apply(
    lambda row: predict_fault(
        row["Voltage"],
        row["Current"],
        row["Temperature"],
        row["Vibration"],
        row["Speed"],
        row["Torque"]
    ),
    axis=1
)

df["Condition"] = df["Condition"].astype(str).str.strip()
df["conventional_Prediction"] = df["conventional_Prediction"].astype(str).str.strip()

correct = (df["Condition"] == df["conventional_Prediction"]).sum()
total = len(df)
wrong = total - correct
unknown_count = (df["conventional_Prediction"] == "Unknown").sum()
accuracy = (correct / total) * 100

print("Total Samples:", total)
print("Correct Predictions:", correct)
print("Wrong Predictions:", wrong)
print("Unknown Predictions:", unknown_count)
print("conventional-Based Accuracy:", round(accuracy, 2), "%")

print("\nconventional Prediction Count:")
print(df["conventional_Prediction"].value_counts())

print("\nClassification Report:")
print(classification_report(
    df["Condition"],
    df["conventional_Prediction"],
    zero_division=0
))

df.to_excel("conventional_Based_Output.xlsx", index=False)

wrong_df = df[df["Condition"] != df["conventional_Prediction"]]
wrong_df.to_excel("conventional_Wrong_Predictions.xlsx", index=False)

graph_df = df[df["conventional_Prediction"] != "Unknown"]


# Feature Graphs
for feature in features:

    plt.figure(figsize=(12, 6))

    avg_values = (
        graph_df.groupby("conventional_Prediction")[feature]
        .mean()
        .reindex(fault_order)
    )

    bars = plt.bar(
        avg_values.index,
        avg_values.values,
        color=colors
    )

    plt.title(
        f"conventional-Based : {feature} under Different Conditions",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel("Predicted Fault Type", fontsize=12)
    plt.ylabel(feature, fontsize=12)

    plt.xticks(rotation=15)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

    # Legend outside graph to avoid overlapping
    plt.legend(
        handles=legend_elements,
        title="Fault Types",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.grid(False)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    max_val = avg_values.max()
    plt.ylim(0, max_val * 1.25)

    plt.tight_layout(rect=[0, 0, 0.82, 1])

    plt.savefig(
        f"conventional_{feature}_Graph.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

# Accuracy Graph - Final Professional Version

plt.figure(figsize=(4.5, 5))

bars = plt.bar(
    ["conventional-Based"],
    [accuracy],
    color="purple",
    width=0.18
)

plt.title(
    "conventional-Based Fault Prediction Accuracy",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel(
    "Method",
    fontsize=12,
    fontweight="bold"
)

plt.ylabel(
    "Accuracy (%)",
    fontsize=12,
    fontweight="bold"
)

plt.xticks(
    fontsize=10,
    fontweight="bold"
)

plt.yticks(
    fontsize=10,
    fontweight="bold"
)

plt.ylim(0, 110)
plt.xlim(-0.4, 0.4)

for bar in bars:
    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 2,
        f"{height:.2f}%",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold"
    )

plt.grid(False)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig(
    "conventional_Accuracy_Graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nconventional-based output, wrong predictions, colored graphs with non-overlapping legend, and accuracy graph generated successfully.")
