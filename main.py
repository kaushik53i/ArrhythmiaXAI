# =========================================================
# ARRHYTHMIAXAI - MAIN PIPELINE
# =========================================================

"""
=========================================================
Author  : Kaushik Bhowmick
Project : ArrhythmiaXAI

Description:
-------------
This file controls the complete machine learning pipeline.

Pipeline Steps:

1. SMOTE Preprocessing
2. Machine Learning Training
3. Model Evaluation
4. TOPSIS Ranking
5. Friedman Statistical Test
6. Wilcoxon Statistical Test
7. SHAP & LIME Explainability

=========================================================
"""

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import subprocess

import time


# =========================================================
# FUNCTION TO RUN SCRIPTS
# =========================================================

def run_script(script_name):

    print("\n" + "=" * 70)

    print(f"🚀 RUNNING: {script_name}")

    print("=" * 70)

    # Start timer
    start_time = time.time()

    # Run script
    result = subprocess.run(

        ["python", script_name],

        text=True
    )

    # End timer
    end_time = time.time()

    execution_time = end_time - start_time

    # -----------------------------------------------------
    # CHECK EXECUTION STATUS
    # -----------------------------------------------------

    if result.returncode == 0:

        print(f"\n✅ COMPLETED: {script_name}")

    else:

        print(f"\n❌ FAILED: {script_name}")

    # -----------------------------------------------------
    # EXECUTION TIME
    # -----------------------------------------------------

    print(
        f"⏱ Execution Time: "
        f"{execution_time:.2f} seconds"
    )


# =========================================================
# MAIN FUNCTION
# =========================================================

def main():

    print("\n" + "#" * 75)

    print(
        "🫀 ARRHYTHMIAXAI - ECG ARRHYTHMIA ANALYSIS PROJECT"
    )

    print("#" * 75)

    print("""

Pipeline Steps:

1. SMOTE Preprocessing
2. Machine Learning Model Training
3. Advanced Model Evaluation
4. TOPSIS Ranking
5. Friedman Statistical Analysis
6. Wilcoxon Statistical Testing
7. SHAP & LIME Explainability

""")


    # =====================================================
    # TOTAL PIPELINE TIMER
    # =====================================================

    total_start_time = time.time()

    # =====================================================
    # RUN ALL PIPELINE SCRIPTS
    # =====================================================

    run_script(
        "scripts/01_Data_Preprocessing.py"
    )

    # -----------------------------------------------------

    run_script(
        "scripts/02_Model_Training.py"
    )

    # -----------------------------------------------------

    # Uncomment after creation
    """
    run_script(
        "scripts/03_Model_Evaluation.py"
    )

    run_script(
        "scripts/04_TOPSIS_Ranking.py"
    )

    run_script(
        "scripts/05_Friedman_Test.py"
    )

    run_script(
        "scripts/06_Wilcoxon_Test.py"
    )

    run_script(
        "scripts/07_SHAP_LIME.py"
    )
    """

    # =====================================================
    # TOTAL EXECUTION TIME
    # =====================================================

    total_end_time = time.time()

    total_execution_time = (

        total_end_time -

        total_start_time
    )

    # =====================================================
    # FINAL MESSAGE
    # =====================================================

    print("\n" + "=" * 75)

    print(
        "🎉 COMPLETE ECG MACHINE LEARNING PIPELINE FINISHED"
    )

    print("=" * 75)

    print(f"""

Total Pipeline Execution Time:
⏱ {total_execution_time:.2f} seconds


Generated Outputs:

✅ Balanced ECG Dataset

✅ Model Training Results

✅ Accuracy Comparison Graphs

✅ SMOTE Visualization

✅ Correlation Heatmap

✅ Feature Analysis


Next Steps:

1. Model Evaluation
2. TOPSIS Ranking
3. Statistical Testing
4. SHAP & LIME Explainability

""")


# =========================================================
# RUN MAIN PIPELINE
# =========================================================

if __name__ == "__main__":

    main()