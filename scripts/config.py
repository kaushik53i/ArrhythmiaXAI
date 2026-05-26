# =========================================================
# ARRHYTHMIAXAI - CONFIGURATION FILE
# =========================================================

"""
=========================================================
This file stores:

1. Dataset Paths
2. Output Paths
3. ML Constants
4. Global Configuration Variables

=========================================================
"""

# =========================================================
# DATASET PATHS
# =========================================================

RAW_DATASET_PATH = (
    "dataset/INCART 2-lead Arrhythmia Database.csv"
)

PROCESSED_DATASET_PATH = (
    "dataset/processed_arrhythmia_data.csv"
)

BALANCED_DATASET_PATH = (
    "dataset/balanced_arrhythmia_data.csv"
)

# =========================================================
# OUTPUT FILE PATHS
# =========================================================

MODEL_RESULTS_PATH = (
    "outputs/model_results.csv"
)

# =========================================================
# RANDOM STATE
# =========================================================

RANDOM_STATE = 42

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

TEST_SIZE = 0.2

# =========================================================
# MODEL PARAMETERS
# =========================================================

MAX_ITER = 1000