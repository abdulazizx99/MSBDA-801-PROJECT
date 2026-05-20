import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("final_project/reports/figures", exist_ok=True)
os.makedirs("final_project/reports/results", exist_ok=True)

results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "Linear SVM"],
    "Accuracy": [0.9137, 0.8689, 0.9367],
    "F1 Score": [0.9137, 0.8689, 0.9367],
    "ROC-AUC": [0.9725, 0.9539, 0.9822]
})

results.to_excel("final_project/reports/results/arabic_ai_results.xlsx", index=False)

plt.figure(figsize=(8,5))
plt.plot(results["Model"], results["Accuracy"], marker="o", label="Accuracy")
plt.plot(results["Model"], results["F1 Score"], marker="o", label="F1 Score")
plt.plot(results["Model"], results["ROC-AUC"], marker="o", label="ROC-AUC")
plt.title("Model Performance Comparison")
plt.xlabel("Model")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.savefig("final_project/reports/figures/model_comparison_chart.png", dpi=300, bbox_inches="tight")
plt.close()

print("Excel and figure saved successfully.")
