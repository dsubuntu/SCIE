import argparse

import numpy as np
import pandas as pd
from openai import OpenAI


def get_LLM_answer(client: OpenAI, model: str, instruction, input):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"{input}\n{instruction}",
            },
        ],
        model=model,
    )
    LLM_answer = response.choices[0].message.content.strip()
    return LLM_answer


def get_numeric_answer(client: OpenAI, model: str, LLM_answer):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"The explanation is: {LLM_answer}\n Extract the answer with Yes/No. Ensure the answer is only one of Yes and No without any punctuation",
            },
        ],
        model=model,
    )
    LLM_numeric_answer = response.choices[0].message.content.strip()
    return LLM_numeric_answer


def process_dataset(
    ds_path: str,
    df: pd.DataFrame,
    client: OpenAI,
    model: str,
):
    question_df = pd.read_json(ds_path)
    question_df = pd.json_normalize(question_df["examples"])

    np.random.seed(43)
    rows_to_exclude = np.random.choice(question_df.index, size=question_df.shape[0] - 5, replace=False)

    question_df = question_df.drop(rows_to_exclude)

    LLM_answers = []

    for _, df_row in df.iterrows():
        for _, question_row in question_df.iterrows():
            instruction = df_row['instruction']
            if "input" in question_row:
                input = question_row['input']
            else:
                input = question_row['question']
            LLM_answer = get_LLM_answer(client, model, instruction, input)
            LLM_answers.append(LLM_answer)

    llm_df = pd.DataFrame({'LLM_answer': LLM_answers})

    repeated_df = pd.DataFrame(np.repeat(df.values, len(question_df), axis=0), columns=df.columns)
    repeated_question_df = pd.concat([question_df] * len(df), ignore_index=True)

    df = pd.concat([repeated_df, repeated_question_df, llm_df], axis=1)

    LLM_numeric_answers = []
    for _, row in df.iterrows():
        LLM_answer = row['LLM_answer']
        LLM_numeric_answer = get_numeric_answer(client, model, LLM_answer)
        LLM_numeric_answers.append(LLM_numeric_answer)

    df['LLM_numeric_answer'] = LLM_numeric_answers

    if "answer" in df.columns:
        df['Y'] = (df['LLM_numeric_answer'] == df['answer']).astype(int)
    else:
        df['Y'] = (df['LLM_numeric_answer'] == df['target']).astype(int)

    return df


def get_num_answer_test(text):
    return text.split()[-1]


def get_numeric_answer2(client, model, LLM_answer):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"The explanation is: {LLM_answer}\nGive the answer in the shortest form possible that will still be correct.Ensure the answer is noly a numeric number.",
            },
        ],
        model=model,
    )
    LLM_numeric_answer = response.choices[0].message.content.strip()
    return LLM_numeric_answer


def process_dataset2(
    ds_path: str,
    df: pd.DataFrame,
    client: OpenAI,
    model: str,
):
    question_df = pd.read_json(ds_path, lines=True)

    question_df["numeric_answer"] = question_df.apply(lambda x: get_num_answer_test(x["answer"]), axis=1)

    np.random.seed(43)
    rows_to_exclude = np.random.choice(question_df.index, size=question_df.shape[0] - 5, replace=False)

    question_df = question_df.drop(rows_to_exclude)

    LLM_answers = []

    for _, df_row in df.iterrows():
        for _, question_row in question_df.iterrows():
            instruction = df_row['instruction']
            input = question_row['question']
            LLM_answer = get_LLM_answer(client, model, instruction, input)
            LLM_answers.append(LLM_answer)

    llm_df = pd.DataFrame({'LLM_answer': LLM_answers})

    repeated_df = pd.DataFrame(np.repeat(df.values, len(question_df), axis=0), columns=df.columns)
    repeated_question_df = pd.concat([question_df] * len(df), ignore_index=True)

    df = pd.concat([repeated_df, repeated_question_df, llm_df], axis=1)

    LLM_numeric_answers = []
    for _, row in df.iterrows():
        LLM_answer = row['LLM_answer']
        LLM_numeric_answer = get_numeric_answer2(client, model, LLM_answer)
        LLM_numeric_answers.append(LLM_numeric_answer)

    df['LLM_numeric_answer'] = LLM_numeric_answers

    df['Y'] = (df['LLM_numeric_answer'] == df['numeric_answer']).astype(int)

    return df


def get_LLM_answer3(client, model, instruction, choices, stem):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"{stem}{choices}{instruction}",
            },
        ],
        model=model,
    )
    LLM_answer = response.choices[0].message.content.strip()
    return LLM_answer


def process_dataset3(
    ds_path: str,
    df: pd.DataFrame,
    client: OpenAI,
    model: str,
):
    question_df = pd.read_json(ds_path, lines=True)

    question_column = pd.json_normalize(question_df["question"])

    question_df = question_df.reset_index(drop=True)
    question_column = question_column.reset_index(drop=True)

    question_df['question_concept'] = question_column['question_concept']
    question_df['stem'] = question_column['stem']
    question_df['choices'] = question_column['choices']

    question_df['choices'] = question_column['choices'].apply(lambda x: ', '.join([f"{choice['label']}: {choice['text']}" for choice in x]))

    np.random.seed(43)
    rows_to_exclude = np.random.choice(question_df.index, size=question_df.shape[0] - 5, replace=False)

    question_df = question_df.drop(rows_to_exclude)

    LLM_answers = []

    for _, df_row in df.iterrows():
        for _, question_row in question_df.iterrows():
            instruction = df_row['instruction']
            stem = question_row['stem']
            choices = question_row['choices']
            LLM_answer = get_LLM_answer3(client, model, instruction, choices, stem)
            LLM_answers.append(LLM_answer)

    llm_df = pd.DataFrame({'LLM_answer': LLM_answers})

    repeated_df = pd.DataFrame(np.repeat(df.values, len(question_df), axis=0), columns=df.columns)
    repeated_question_df = pd.concat([question_df] * len(df), ignore_index=True)

    df = pd.concat([repeated_df, repeated_question_df, llm_df], axis=1)

    LLM_numeric_answers = []
    for _, row in df.iterrows():
        LLM_answer = row['LLM_answer']
        LLM_numeric_answer = get_numeric_answer(client, model, LLM_answer)
        LLM_numeric_answers.append(LLM_numeric_answer)

    df['LLM_numeric_answer'] = LLM_numeric_answers

    df['Y'] = (df['LLM_numeric_answer'] == df['answerKey']).astype(int)

    return df


def get_chatgpt_answer(client, model, question, StepInstruct, LLM_step):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"{question}\n {StepInstruct}\n {LLM_step} \n Therefore, the answer is which one of (A)/(B)/(C)/(D)/(E)/(F)?",
            }
        ],
        model=model,
    )
    answer = response.choices[0].message.content.strip()
    return answer


def extract_numeric_answer(client, model, LLM_answer):
    prompt = f"The explanation is: {LLM_answer}\n Extract the answer with only one of (A), (B), (C), (D), (E), (F) without the option's content (only a bracket and a letter). Show the answer without any preparatory statements"

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
    )
    clean_answer = response.choices[0].message.content.strip()
    return clean_answer


def process_dataset4(
    ds_path: str,
    df: pd.DataFrame,
    client: OpenAI,
    model: str,
):
    question_df = pd.read_json(ds_path)
    question_df = pd.json_normalize(question_df["examples"])

    np.random.seed(43)
    rows_to_exclude = np.random.choice(question_df.index, size=question_df.shape[0] - 5, replace=False)

    question_df = question_df.drop(rows_to_exclude)

    LLM_answers = []

    for _, df_row in df.iterrows():
        for _, question_row in question_df.iterrows():
            instruction = df_row['instruction']
            input = question_row['input']
            LLM_answer = get_LLM_answer(client, model, instruction, input)
            LLM_answers.append(LLM_answer)

    llm_df = pd.DataFrame({'LLM_answer': LLM_answers})

    repeated_df = pd.DataFrame(np.repeat(df.values, len(question_df), axis=0), columns=df.columns)
    repeated_question_df = pd.concat([question_df] * len(df), ignore_index=True)

    df = pd.concat([repeated_df, repeated_question_df, llm_df], axis=1)

    LLM_answers = []
    for idx, row in df.iterrows():
        question = row['input']
        StepInstruct = row['instruction']
        LLM_step = row['LLM_step']
        LLM_answer = get_chatgpt_answer(client, model, question, StepInstruct, LLM_step)

        LLM_answers.append(LLM_answer)

    df['LLM_answer'] = LLM_answers

    LLM_numeric_answers = []
    for idx, row in df.iterrows():
        LLM_answer = row['LLM_answer']
        LLM_numeric_answer = extract_numeric_answer(client, model, LLM_answer)

        LLM_numeric_answers.append(LLM_numeric_answer)

    df['LLM_numeric_answer'] = LLM_numeric_answers

    df['Y'] = (df['LLM_numeric_answer'] == df['target']).astype(int)

    return df


def process_dataset5(
    ds_path: str,
    df: pd.DataFrame,
    client: OpenAI,
    model: str,
):
    question_df = pd.read_json(ds_path)

    np.random.seed(43)
    rows_to_exclude = np.random.choice(question_df.index, size=question_df.shape[0] - 5, replace=False)

    question_df = question_df.drop(rows_to_exclude)

    LLM_answers = []

    for _, df_row in df.iterrows():
        for _, question_row in question_df.iterrows():
            instruction = df_row['instruction']
            if "input" in question_row:
                input = question_row['input']
            elif "sQuestion" in question_row:
                input = question_row['sQuestion']
            else:
                input = question_row['question']
            LLM_answer = get_LLM_answer(client, model, instruction, input)
            LLM_answers.append(LLM_answer)

    llm_df = pd.DataFrame({'LLM_answer': LLM_answers})

    repeated_df = pd.DataFrame(np.repeat(df.values, len(question_df), axis=0), columns=df.columns)
    repeated_question_df = pd.concat([question_df] * len(df), ignore_index=True)

    df = pd.concat([repeated_df, repeated_question_df, llm_df], axis=1)

    LLM_numeric_answers = []
    for _, row in df.iterrows():
        LLM_answer = row['LLM_answer']
        LLM_numeric_answer = get_numeric_answer(client, model, LLM_answer)
        LLM_numeric_answers.append(LLM_numeric_answer)

    df['LLM_numeric_answer'] = LLM_numeric_answers

    df['lSolutions'] = df['lSolutions'].astype(str)
    df['lSolutions'] = df['lSolutions'].str.replace(r'[\[\]]', '', regex=True)
    df['lSolutions'] = df['lSolutions'].str.strip("'")

    df['LLM_numeric_answer'] = pd.to_numeric(df['LLM_numeric_answer'], errors='coerce')
    df['lSolutions'] = pd.to_numeric(df['lSolutions'], errors='coerce')

    if "answer" in df.columns:
        df['Y'] = (df['LLM_numeric_answer'] == df['answer']).astype(int)
    elif "lSolutions" in df.columns:
        df['Y'] = (df['LLM_numeric_answer'] == df['lSolutions']).astype(int)
    else:
        df['Y'] = (df['LLM_numeric_answer'] == df['target']).astype(int)

    return df


def extract_unique_value(score_dict):
    for key, value in score_dict.items():
        if value == 1:
            return key
    return None


def process_dataset6(
    ds_path: str,
    df: pd.DataFrame,
    client: OpenAI,
    model: str,
):
    question_df = pd.read_json(ds_path)

    question_df['answer'] = question_df['target_scores'].apply(extract_unique_value)

    np.random.seed(43)
    rows_to_exclude = np.random.choice(question_df.index, size=question_df.shape[0] - 5, replace=False)

    question_df = question_df.drop(rows_to_exclude)

    LLM_answers = []

    for _, df_row in df.iterrows():
        for _, question_row in question_df.iterrows():
            instruction = df_row['instruction']
            if "input" in question_row:
                input = question_row['input']
            elif "sQuestion" in question_row:
                input = question_row['sQuestion']
            else:
                input = question_row['question']
            LLM_answer = get_LLM_answer(client, model, instruction, input)
            LLM_answers.append(LLM_answer)

    llm_df = pd.DataFrame({'LLM_answer': LLM_answers})

    repeated_df = pd.DataFrame(np.repeat(df.values, len(question_df), axis=0), columns=df.columns)
    repeated_question_df = pd.concat([question_df] * len(df), ignore_index=True)

    df = pd.concat([repeated_df, repeated_question_df, llm_df], axis=1)

    LLM_numeric_answers = []
    for _, row in df.iterrows():
        LLM_answer = row['LLM_answer']
        LLM_numeric_answer = get_numeric_answer(client, model, LLM_answer)
        LLM_numeric_answers.append(LLM_numeric_answer)

    df['LLM_numeric_answer'] = LLM_numeric_answers

    if "answer" in df.columns:
        df['Y'] = (df['LLM_numeric_answer'] == df['answer']).astype(int)
    elif "lSolutions" in df.columns:
        df['Y'] = (df['LLM_numeric_answer'] == df['lSolutions']).astype(int)
    else:
        df['Y'] = (df['LLM_numeric_answer'] == df['target']).astype(int)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Dataset to process",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="OpenAI API key",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://api.openai.com/v1/chat",
        help="Base URL for the OpenAI API",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use for evaluation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_observational_dataset",
        help="Path to the output file",
    )
    args = parser.parse_args()

    df = pd.read_csv("./instructions_data.csv")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    datasets = []
    if args.dataset == "all":
        datasets = [
            "boolean_expressions",
            "causal_judgement",
            "coin_flip",
            "CommonsenseQA",
            "date_understanding",
            "disambiguation_qa",
            "GSM8K",
            "last_letters",
            "MultiArith",
            "StrategyQA",
        ]
    elif args.dataset == "causal_judgement":
        datasets = ["causal_judgement"]
    elif args.dataset == "boolean_expressions":
        datasets = ["boolean_expressions"]
    elif args.dataset == "coin_flip":
        datasets = ["coin_flip"]
    elif args.dataset == "CommonsenseQA":
        datasets = ["CommonsenseQA"]
    elif args.dataset == "date_understanding":
        datasets = ["date_understanding"]
    elif args.dataset == "disambiguation_qa":
        datasets = ["disambiguation_qa"]
    elif args.dataset == "GSM8K":
        datasets = ["GSM8K"]
    elif args.dataset == "last_letters":
        datasets = ["last_letters"]
    elif args.dataset == "MultiArith":
        datasets = ["MultiArith"]
    elif args.dataset == "StrategyQA":
        datasets = ["StrategyQA"]
    else:
        raise ValueError(f"Dataset {args.dataset} is not supported yet.")

    ds_paths = {
        "causal_judgement": "./input-dataset/causal_judgement.json",
        "boolean_expressions": "./input-dataset/boolean_expressions.json",
        "coin_flip": "./input-dataset/coin_flip.json",
        "CommonsenseQA": "./input-dataset/CommonsenseQA.jsonl",
        "date_understanding": "./input-dataset/date_understanding.json",
        "disambiguation_qa": "./input-dataset/disambiguation_qa.json",
        "GSM8K": "./input-dataset/GSM8K.jsonl",
        "last_letters": "./input-dataset/last_letters.json",
        "MultiArith": "./input-dataset/MultiArith.json",
        "StrategyQA": "./input-dataset/StrategyQA.json",
    }

    for dataset in datasets:
        ds_path = ds_paths[dataset]
        print(f"Processing dataset: {dataset}")

        if "GSM8K" in ds_path:
            df = process_dataset2(ds_path, df, client=client, model=args.model)
        elif "CommonsenseQA" in ds_path:
            df = process_dataset3(ds_path, df, client=client, model=args.model)
        elif "date_understanding" in ds_path:
            df = process_dataset4(ds_path, df, client=client, model=args.model)
        elif "disambiguation_qa" in ds_path:
            df = process_dataset4(ds_path, df, client=client, model=args.model)
        elif "MultiArith" in ds_path:
            df = process_dataset5(ds_path, df, client=client, model=args.model)
        else:
            df = process_dataset(ds_path, df, client=client, model=args.model)

        save_path = f"{args.output_dir}/{dataset}-observational_data.csv"
        df.to_csv(save_path, index=False, sep=",")


if __name__ == "__main__":
    main()
