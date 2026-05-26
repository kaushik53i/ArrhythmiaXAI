# =========================================================
# ARRHYTHMIAXAI - EVALUATION PLOTS
# =========================================================

"""
=========================================================
This file contains:

1. Confusion Matrix Plot
2. ROC Curve Plot

=========================================================
"""

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import confusion_matrix


# =========================================================
# CONFUSION MATRIX
# =========================================================

def plot_confusion_matrix(

    y_test,

    y_pred,

    model_name
):

    cm = confusion_matrix(

        y_test,

        y_pred
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(

        cm,

        annot=True,

        fmt='d',

        cmap='Blues'
    )

    plt.title(
        f"{model_name} Confusion Matrix"
    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.savefig(

        f"outputs/graphs/confusion_matrix_{model_name}.png",

        bbox_inches="tight"
    )

    plt.close()