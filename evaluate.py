import argparse

import numpy as np
import pandas as pd
from openai import OpenAI


class Evaluator:
    def __init__(
        self,
        step_instruction: str,
        api_key: str,
        base_url: str,
        model: str = "gpt-3.5-turbo",
    ):
        self.step_instruction = step_instruction

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        self.model = model

    def get_step_answer(self, question: str) -> str:
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"{question}\n {self.step_instruction}\n",
                }
            ],
            model=self.model,
        )

        answer = response.choices[0].message.content.strip()
        return answer

    def get_chatgpt_answer(self, question: str, llm_step: str) -> str:
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"{question}\n {self.step_instruction}\n {llm_step} \n Therefore, the answer is",
                }
            ],
            model=self.model,
        )
        answer = response.choices[0].message.content.strip()
        return answer

    def extract_numeric_answer(self, llm_step_instruct_answer: str) -> str:
        prompt = f"The answer is: {llm_step_instruct_answer}\nExtract the answer with Yes/No. Ensure the answer is only one of Yes and No without any punctuation"

        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model,
        )
        gpt4_numeric_answer = response.choices[0].message.content.strip()
        return gpt4_numeric_answer

    def extract_numeric_answer2(self, LLM_StepInstruct_answer):
        prompt = f"The explanation is: {LLM_StepInstruct_answer}\nGive the answer in the shortest form possible that will still be correct.Ensure the answer is noly a numeric number."

        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model,
        )
        gpt4_numeric_answer = response.choices[0].message.content.strip()
        return gpt4_numeric_answer

    def extract_numeric_answer3(self, LLM_StepInstruct_answer):
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"The explanation is: {LLM_StepInstruct_answer}\n Extract the answer with A/B/C/D/E. Ensure the answer is only one of A/B/C/D/E without any punctuation",
                },
            ],
            model=self.model,
        )
        gpt4_numeric_answer = response.choices[0].message.content.strip()
        return gpt4_numeric_answer

    def evaluate(self, df: pd.DataFrame) -> float:
        llm_steps = []

        for _, row in df.iterrows():
            if "stem" in row:
                question = row["stem"] + " " + row["choices"]
            elif "input" in row:
                question = row["input"]
            elif "sQuestion" in row:
                question = row["sQuestion"]
            else:
                question = row["question"]
            step = self.get_step_answer(question)

            llm_steps.append(step)

        df["LLM_step"] = llm_steps

        llm_step_instruct_answers = []
        for _, row in df.iterrows():
            if "stem" in row:
                question = row["stem"] + " " + row["choices"]
            elif "input" in row:
                question = row["input"]
            elif "sQuestion" in row:
                question = row["sQuestion"]
            else:
                question = row["question"]
            llm_step = row["LLM_step"]
            answer = self.get_chatgpt_answer(question, llm_step)
            llm_step_instruct_answers.append(answer)

        df["LLM_StepInstruct_answer"] = llm_step_instruct_answers

        llm_step_instruct_numeric_answers = []
        for _, row in df.iterrows():
            llm_step_instruct_answer = row["LLM_StepInstruct_answer"]
            if "numeric_answer" in row:
                LLM_StepInstruct_numeric_answer = self.extract_numeric_answer2(llm_step_instruct_answer)
            elif "stem" in row:
                LLM_StepInstruct_numeric_answer = self.extract_numeric_answer3(llm_step_instruct_answer)
            else:
                LLM_StepInstruct_numeric_answer = self.extract_numeric_answer(llm_step_instruct_answer)

            llm_step_instruct_numeric_answers.append(LLM_StepInstruct_numeric_answer)

        df["LLM_StepInstruct_numeric_answer"] = llm_step_instruct_numeric_answers

        if "target" in df.columns:
            df["LLM_StepInstruct_binary_result"] = (df["LLM_StepInstruct_numeric_answer"] == df["target"]).astype(int)
        elif "answer" in df.columns:
            df["LLM_StepInstruct_binary_result"] = (df["LLM_StepInstruct_numeric_answer"] == df["answer"]).astype(int)
        elif "numeric_answer" in df.columns:
            df["LLM_StepInstruct_binary_result"] = (df["LLM_StepInstruct_numeric_answer"] == df["numeric_answer"]).astype(int)
        elif "answerKey" in df.columns:
            df["LLM_StepInstruct_binary_result"] = (df["LLM_StepInstruct_numeric_answer"] == df["answerKey"]).astype(int)
        elif "lSolutions" in df.columns:
            df['lSolutions'] = df['lSolutions'].astype(str)
            df['lSolutions'] = df['lSolutions'].str.replace(r'[\[\]]', '', regex=True)
            df['LLM_StepInstruct_numeric_answer'] = pd.to_numeric(df['LLM_StepInstruct_numeric_answer'], errors='coerce')
            df['lSolutions'] = pd.to_numeric(df['lSolutions'], errors='coerce')
            df['LLM_StepInstruct_binary_result'] = (df['LLM_StepInstruct_numeric_answer'] == df['lSolutions']).astype(int)

        acc = df["LLM_StepInstruct_binary_result"].mean()

        return float(acc)


def extract_unique_value(score_dict):
    for key, value in score_dict.items():
        if value == 1:
            return key
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--enh_instruction",
        type=str,
        default="Ensure clarity and maintain a positive, engaging tone. Proceed methodically, articulating each thought step by step with clear, precise language.",
    )
    parser.add_argument(
        "--base_instruction",
        type=str,
        default="Let's think step by step.",
        help="Base instruction for generating instructions",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="./input-dataset/causal_judgement.json",
        help="Path to the data file",
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
        default="gpt-3.5-turbo",
        help="Model to use for evaluation",
    )
    args = parser.parse_args()

    if ".jsonl" in args.data:
        df = pd.read_json(args.data, lines=True)
    else:
        df = pd.read_json(args.data)

    if "StrategyQA" in args.data:
        df['answer'] = df['target_scores'].apply(extract_unique_value)
    elif "GSM8K" in args.data:
        def get_num_answer_test(text):
            return text.split()[-1]
        df["numeric_answer"] = df.apply(lambda x: get_num_answer_test(x["answer"]), axis=1)
    elif "CommonsenseQA" in args.data:
        test_column = pd.json_normalize(test['question'])
        test = test.reset_index(drop=True)
        test_column = test_column.reset_index(drop=True)
        test['question_concept'] = test_column['question_concept']
        test['stem'] = test_column['stem']
        test['choices'] = test_column['choices']
        test['choices'] = test_column['choices'].apply(lambda x: ', '.join([f"{choice['label']}: {choice['text']}" for choice in x]))
        df = test
    else:
        df = pd.json_normalize(df["examples"])

    np.random.seed(43)
    rows_to_include = np.random.choice(df.index, size=5, replace=False)

    filtered_df: pd.DataFrame = df.drop(rows_to_include)

    evaluator = Evaluator(
        step_instruction=args.base_instruction,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
    )

    acc = evaluator.evaluate(filtered_df.copy())
    print(f"BaseInstruction Accuracy: {acc}")

    evaluator = Evaluator(
        step_instruction=args.enh_instruction,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
    )

    acc = evaluator.evaluate(filtered_df.copy())
    print(f"EnhancedInstruction Accuracy: {acc}")


if __name__ == "__main__":
    main()
