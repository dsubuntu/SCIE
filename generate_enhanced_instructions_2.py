import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from causalnlp import CausalInferenceModel
from interpreter import interpreter


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
        default="./generated_observational_dataset/causal_judgement-structural_data.csv",
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
    parser.add_argument(
        "--api_key",
        type=str,
        help="OpenAI API key",
    )

    args = parser.parse_args()

    interpreter.llm.api_key = args.api_key
    interpreter.auto_run = True

    interpreter.messages = []

    data_path = Path(args.data)
    file_name = data_path.name

    res = interpreter.chat(
        f"""Find the file '{file_name}' located in "{str(data_path)}" and list its columns."""
    )
    print(res)

    res = interpreter.chat("""For above file 'causal_judgement-causal_data.csv', write code to calculate the Average Treatment Effect (ATE) of each treatment, recursively. The treatment is 2 to 9 column (the treatment should be binarized according to the mean of column before being treated as a treatment) in turn, the covariates is the other columns in 2 to 9 columns except the current treatment, and the outcome is the column 'Y'.
Here is the reference code:
import numpy as np
from causalnlp import CausalInferenceModel

def binarize_column(data, column):
    mean_value = data[column].mean()
    return np.where(data[column] > mean_value, 1, 0)

results = []

for col_index in range(1, 9):
    treatment_col = df.columns[col_index]

    df_copy = df.copy()
    df_copy[treatment_col] = binarize_column(df_copy, treatment_col)

    cm = CausalInferenceModel(df_copy,
                              metalearner_type='t-learner',
                              treatment_col=treatment_col,
                              outcome_col='Y',
                              ignore_cols=['instruction', 'input','target', 'answer-num','LLM_answer','LLM_answer', 'LLM_numeric_answer']).fit()

    ate = cm.estimate_ate()
    results.append(ate)

for idx, result in enumerate(results):
    print(f"ATE for treatment column {df.columns[idx + 1]}: {result}")

Some of the reference answers are:
ATE for treatment column Clarity: {'ate': 0.21176574164629985}
ATE for treatment column Action-Oriented: {'ate': 0.14324372325019052}
ATE for treatment column Specificity: {'ate': 0.227679163221092}
ATE for treatment column Conciseness: {'ate': 0.04663181467360423}

Complete all answers at the end.""")
    print(res)

    res = interpreter.chat("""To let the large language model complete the task, please generate a new LLM input instruction, making the overall ATE the most, based on the following example: "Let's think step by step.".""")
    print(res)


if __name__ == "__main__":
    main()
