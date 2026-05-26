# =========================================================
# ARRHYTHMIAXAI - ULTRA FAST SMOTE PREPROCESSING
# =========================================================

"""
=========================================================
Author  : Kaushik Bhowmick
Project : ArrhythmiaXAI

Description:
-------------
Ultra fast preprocessing pipeline optimized for:
✅ Large ECG datasets
✅ Low RAM usage
✅ Faster SMOTE execution
✅ Faster training preparation
✅ Automatic graph saving
✅ Production-level workflow

=========================================================
"""

# =========================================================
# IMPORT LIBRARIES
# =========================================================

# Data handling
import pandas as pd

# Visualization
import matplotlib.pyplot as plt

# Train-test split
from sklearn.model_selection import train_test_split

# SMOTE
from imblearn.over_sampling import SMOTE

# System utilities
import os
import time

# =========================================================
# IMPORT CUSTOM MODULES
# =========================================================

from config import *
from utils import *

# =========================================================
# CREATE OUTPUT DIRECTORIES
# =========================================================

os.makedirs(
    "outputs",
    exist_ok=True
)

os.makedirs(
    "outputs/graphs",
    exist_ok=True
)

# =========================================================
# CLASS DISTRIBUTION GRAPH
# =========================================================

def save_class_distribution_graph(target_data):

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

    # Faster than plt.show()
    plt.close()


# =========================================================
# BALANCED DATASET GRAPH
# =========================================================

def save_balanced_dataset_graph(target_data):

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

    # Faster than plt.show()
    plt.close()


# =========================================================
# MAIN FUNCTION
# =========================================================

def main():

    # =====================================================
    # START TIMER
    # =====================================================

    start_time = time.time()

    # =====================================================
    # STEP TITLE
    # =====================================================

    print_step(
        1,
        "ULTRA FAST SMOTE PREPROCESSING"
    )

    # =====================================================
    # LOAD DATASET
    # =====================================================

    print_heading(
        "📂 LOADING DATASET"
    )

    try:

        # Faster loading
        df = pd.read_csv(
            RAW_DATASET_PATH,
            low_memory=True
        )

        print_success(
            "Dataset Loaded Successfully"
        )

    except FileNotFoundError:

        print_error(
            "Dataset File Not Found"
        )

        return

    # =====================================================
    # MEMORY OPTIMIZATION
    # =====================================================

    print_heading(
        "⚡ MEMORY OPTIMIZATION"
    )

    # Convert float64 → float32
    float_columns = df.select_dtypes(
        include=['float64']
    ).columns

    df[float_columns] = df[
        float_columns
    ].astype('float32')

    print_success(
        "Memory Optimization Completed"
    )

    # =====================================================
    # TARGET CLASS DISTRIBUTION
    # =====================================================

    target_column = df.columns[-1]

    print_heading(
        "🎯 TARGET CLASS DISTRIBUTION"
    )

    print(
        df[target_column].value_counts()
    )

    # =====================================================
    # SAVE ORIGINAL DISTRIBUTION GRAPH
    # =====================================================

    save_class_distribution_graph(
        df[target_column]
    )

    print_success(
        "Original Distribution Graph Saved"
    )

    # =====================================================
    # FEATURE & TARGET SEPARATION
    # =====================================================

    X = df.iloc[:, :-1]

    y = df.iloc[:, -1]

    print_success(
        "Features and Target Separated"
    )

    # =====================================================
    # TRAIN TEST SPLIT
    # =====================================================

    print_heading(
        "✂️ TRAIN TEST SPLIT"
    )

    # Stratified split preserves class balance

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y
    )

    print_success(
        "Train-Test Split Completed"
    )

    # =====================================================
    # ULTRA FAST SMOTE
    # =====================================================

    print_heading(
        "⚙️ APPLYING FAST SMOTE"
    )

    # Faster partial balancing
    smote = SMOTE(

        sampling_strategy=0.5,

        random_state=RANDOM_STATE,

        k_neighbors=3,

        n_jobs=-1
    )

    X_train_smote, y_train_smote = smote.fit_resample(

        X_train,
        y_train
    )

    print_success(
        "Fast SMOTE Applied Successfully"
    )

    # =====================================================
    # SAVE BALANCED GRAPH
    # =====================================================

    save_balanced_dataset_graph(
        y_train_smote
    )

    print_success(
        "Balanced Dataset Graph Saved"
    )

    # =====================================================
    # SAVE BALANCED TRAINING DATASET
    # =====================================================

    print_heading(
        "💾 SAVING BALANCED DATASET"
    )

    balanced_df = pd.DataFrame(

        X_train_smote,

        columns=X.columns
    )

    balanced_df[target_column] = y_train_smote

    balanced_df.to_csv(

        BALANCED_DATASET_PATH,

        index=False
    )

    print_success(
        "Balanced Dataset Saved"
    )

    # =====================================================
    # SAVE TEST DATASET
    # =====================================================

    print_heading(
        "💾 SAVING TEST DATASET"
    )

    test_df = pd.DataFrame(

        X_test,

        columns=X.columns
    )

    test_df[target_column] = y_test

    test_df.to_csv(

        "dataset/test_arrhythmia_data.csv",

        index=False
    )

    print_success(
        "Testing Dataset Saved"
    )

    # =====================================================
    # EXECUTION TIME
    # =====================================================

    end_time = time.time()

    execution_time = end_time - start_time

    # =====================================================
    # FINAL MESSAGE
    # =====================================================

    print("\n" + "=" * 60)

    print(
        "🎉 ULTRA FAST SMOTE PREPROCESSING COMPLETED"
    )

    print("=" * 60)

    print(f"""

⏱ Total Execution Time:
{execution_time:.2f} seconds


Generated Outputs:

1. dataset/balanced_arrhythmia_data.csv

2. dataset/test_arrhythmia_data.csv

3. outputs/graphs/class_distribution_before_smote.png

4. outputs/graphs/balanced_dataset.png

""")

# =========================================================
# RUN MAIN FUNCTION
# =========================================================

if __name__ == "__main__":

    main()