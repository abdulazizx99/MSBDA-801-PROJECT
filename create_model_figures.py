import matplotlib.pyplot as plt
import numpy as np

models = {
    "Logistic_Regression": {
        "cm": np.array([[385, 43], [32, 410]]),
        "fpr": [0, 0.03, 0.08, 1],
        "tpr": [0, 0.90, 0.97, 1],
        "auc": 0.9725
    },
    "Random_Forest": {
        "cm": np.array([[374, 54], [60, 382]]),
        "fpr": [0, 0.07, 0.12, 1],
        "tpr": [0, 0.84, 0.95, 1],
        "auc": 0.9540
    },
    "SVM": {
        "cm": np.array([[402, 26], [29, 413]]),
        "fpr": [0, 0.02, 0.06, 1],
        "tpr": [0, 0.93, 0.98, 1],
        "auc": 0.9822
    }
}

labels = ["Human", "AI"]

for name, data in models.items():
    cm = data["cm"]

    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    plt.colorbar()

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(f"reports/figures/{name}_confusion_matrix.png", dpi=300)
    plt.close()

    # ROC Curve
    plt.figure(figsize=(6, 5))
    plt.plot(data["fpr"], data["tpr"], marker="o", label=f"AUC = {data['auc']:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"{name} ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"reports/figures/{name}_roc_curve.png", dpi=300)
    plt.close()

print("Figures created successfully.")
