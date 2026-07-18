from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = BASE_DIR / "reports" / "results"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

model_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "CNN",
        "LSTM",
        "XGBoost Structured Only",
        "Transformer + XGBoost + MLP Fusion"
    ],
    "Model Type": [
        "Classical ML Baseline",
        "Classical ML Baseline",
        "Deep Learning Baseline",
        "Deep Learning Baseline",
        "Structured Feature Model",
        "Hybrid Model"
    ],
    "Dataset": [
        "V2 Multilingual",
        "V2 Multilingual",
        "V2 Multilingual",
        "V2 Multilingual",
        "Hybrid V2 Subset / Structured Features",
        "Hybrid V2 Transformer + Structured Features"
    ],
    "Test Accuracy": [
        98.01,
        98.28,
        97.77,
        87.82,
        80.08,
        95.82
    ],
    "Macro F1": [
        0.980,
        0.984,
        0.979,
        0.886,
        0.819,
        0.948
    ],
    "Remarks": [
        "Strong TF-IDF baseline",
        "Best overall model",
        "Strong deep learning baseline",
        "Lower performance on spam/legitimate separation",
        "Uses only structured features",
        "Completed proposed hybrid architecture"
    ]
})

output_csv = RESULTS_DIR / "final_model_comparison_table.csv"
model_results.to_csv(output_csv, index=False, encoding="utf-8-sig")

print("Final model comparison table saved to:", output_csv)
print(model_results)

plt.figure(figsize=(12, 6))
plt.bar(model_results["Model"], model_results["Test Accuracy"])
plt.title("Final Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Test Accuracy (%)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

chart_path = FIGURES_DIR / "final_model_accuracy_comparison.png"
plt.savefig(chart_path, dpi=300)
plt.show()

print("Model comparison chart saved to:", chart_path)