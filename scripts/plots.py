# =========================================================
# ARRHYTHMIAXAI - VISUALIZATION FUNCTIONS
# =========================================================

"""
=========================================================
Optimized visualization functions for:

1. Class Distribution
2. Balanced Dataset Visualization
3. Correlation Heatmap
4. Accuracy Comparison

=========================================================
"""

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import matplotlib.pyplot as plt

import seaborn as sns

import os

# =========================================================
# CREATE GRAPH DIRECTORY
# =========================================================

os.makedirs(
    "outputs/graphs",
    exist_ok=True
)

# =========================================================
# ORIGINAL CLASS DISTRIBUTION
# =========================================================

def plot_class_distribution(target_data):

    plt.figure(figsize=(8, 5))

    target_data.value_counts().plot(
        kind='bar'
    )

    plt.title(
        "Class Distribution Before SMOTE"
    )

    plt.xlabel("Class")

    plt.ylabel("Count")

    plt.savefig(

        "outputs/graphs/class_distribution_before_smote.png",

        bbox_inches="tight"
    )

    # IMPORTANT:
    # Close graph instead of showing
    plt.close()


# =========================================================
# BALANCED DATASET DISTRIBUTION
# =========================================================

def plot_balanced_dataset(target_data):

    plt.figure(figsize=(8, 5))

    target_data.value_counts().plot(
        kind='bar'
    )

    plt.title(
        "Balanced Dataset After SMOTE"
    )

    plt.xlabel("Class")

    plt.ylabel("Count")

    plt.savefig(

        "outputs/graphs/balanced_dataset.png",

        bbox_inches="tight"
    )

    plt.close()


# =========================================================
# OPTIMIZED CORRELATION HEATMAP
# =========================================================

def plot_correlation_heatmap(df):

    print("\n⏳ Preparing optimized heatmap...")

    # -----------------------------------------------------
    # Select numeric columns only
    # -----------------------------------------------------

    numeric_df = df.select_dtypes(
        include=['number']
    )

    # -----------------------------------------------------
    # Reduce dataset size
    # -----------------------------------------------------

    if len(numeric_df) > 1000:

        numeric_df = numeric_df.sample(

            n=1000,

            random_state=42
        )

    # -----------------------------------------------------
    # Reduce number of columns
    # -----------------------------------------------------

    numeric_df = numeric_df.iloc[:, :15]

    # -----------------------------------------------------
    # Correlation matrix
    # -----------------------------------------------------

    correlation_matrix = numeric_df.corr()

    # -----------------------------------------------------
    # Plot heatmap
    # -----------------------------------------------------

    plt.figure(figsize=(10, 6))

    sns.heatmap(

        correlation_matrix,

        cmap="coolwarm",

        annot=False
    )

    plt.title(
        "Feature Correlation Heatmap"
    )

    plt.savefig(

        "outputs/graphs/correlation_heatmap.png",

        bbox_inches="tight"
    )

    plt.close()


# =========================================================
# MODEL ACCURACY COMPARISON
# =========================================================

def plot_accuracy_comparison(results_df):

    plt.figure(figsize=(14, 7))

    plt.bar(

        results_df["Model"],

        results_df["Accuracy"]
    )

    plt.xticks(rotation=45)

    plt.title(
        "Model Accuracy Comparison"
    )

    plt.xlabel("Models")

    plt.ylabel("Accuracy")

    plt.savefig(

        "outputs/graphs/accuracy_comparison.png",

        bbox_inches="tight"
    )

    plt.close()