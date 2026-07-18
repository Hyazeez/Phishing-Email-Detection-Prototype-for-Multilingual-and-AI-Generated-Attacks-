from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = BASE_DIR / "reports" / "results"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

robustness_results = pd.DataFrame({
    "Model": [
        "MLP Fusion Hybrid"
    ],
    "Test Set": [
        "50 manually designed robustness samples"
    ],
    "Attack Types": [
        "Code-mixed Sinhala-English, Tamil-English, AI-generated phishing, obfuscated URLs, BEC, spam, legitimate multilingual emails"
    ],
    "Robustness Accuracy": [
        72.00
    ],
    "Macro F1": [
        0.43
    ],
    "Key Finding": [
        "Strong on code-mixed and obfuscated URL phishing; weak on AI-generated phishing and BEC classification"
    ]
})

output_csv = RESULTS_DIR / "robustness_testing_summary_table.csv"
robustness_results.to_csv(output_csv, index=False, encoding="utf-8-sig")

print("Robustness testing summary saved to:", output_csv)
print(robustness_results)

plt.figure(figsize=(7, 5))
plt.bar(robustness_results["Model"], robustness_results["Robustness Accuracy"])
plt.title("Robustness Testing Accuracy")
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.tight_layout()

chart_path = FIGURES_DIR / "robustness_testing_accuracy.png"
plt.savefig(chart_path, dpi=300)
plt.show()

print("Robustness chart saved to:", chart_path)