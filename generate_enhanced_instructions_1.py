import argparse

import numpy as np
import pandas as pd
from causalnlp import CausalInferenceModel


def binarize_column(data, column):
    mean_value = data[column].mean()
    return np.where(data[column] > mean_value, 1, 0)


def calculate_ate(df: pd.DataFrame, col_range: range, ignore_cols: list[str]):
    results = []
    for col_index in col_range:
        treatment_col = df.columns[col_index]

        df_copy = df.copy()
        df_copy[treatment_col] = binarize_column(df_copy, treatment_col)

        cm = CausalInferenceModel(
            df_copy,
            metalearner_type="t-learner",
            treatment_col=treatment_col,
            outcome_col="Y",
            ignore_cols=ignore_cols,
        ).fit()

        ate = cm.estimate_ate()
        results.append(ate)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="./generated_observational_dataset/causal_judgement-observational_data.csv",
        help="Path to the data file",
    )
    parser.add_argument(
        "--col_range",
        type=str,
        default="1,9",
        help="Range of columns to calculate ATE for",
    )
    parser.add_argument(
        "--ignore_cols",
        type=str,
        default="instruction,input,target,answer-num,LLM_answer,LLM_numeric_answer",
        help="Columns to ignore",
    )

    args = parser.parse_args()

    # Load the data
    df = pd.read_csv(args.data)

    col_range = list(map(int, args.col_range.split(",")))
    assert len(col_range) == 2
    col_range = range(col_range[0], col_range[1])

    ignore_cols = args.ignore_cols.split(",")

    # Calculate the ATE
    results = calculate_ate(df, col_range, ignore_cols)
    print(results)


if __name__ == "__main__":
    main()
