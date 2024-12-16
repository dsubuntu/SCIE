import argparse

import pandas as pd
from openai import OpenAI


class DataPipeline:
    def __init__(
        self,
        base_instruction: str,
        api_key: str,
        base_url: str,
        model: str = "gpt-4o-mini",
    ):
        self.base_instruction = base_instruction

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        self.model = model

    def generate_instructions(
        self,
        num_instructions: int,
    ) -> list[str]:
        base_instruction = self.base_instruction
        instructions = []

        for _ in range(num_instructions - 1):
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "To let the large language model complete the task of solving problems, "
                            f"please generate a different LLM input instruction, similar to the following example: {base_instruction}. "
                            "Give the answer directly without preparatory statements!"
                        ),
                    }
                ],
                model=self.model,
            )
            sample = response.choices[0].message.content.strip()
            instructions.append(sample)

        instructions.append(base_instruction)
        return instructions

    def extract_features(self, instructions: list[str]) -> str:
        base_instruction = self.base_instruction

        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"To let the large language model complete the task of solving problems, please generate a different LLM input instruction, similar to the following example: {base_instruction}. Give the answer directly without preparatory statements!",
                },
                {
                    "role": "assistant",
                    "content": str(instructions),
                },
                {
                    "role": "user",
                    "content": f"What are the measurable and improveable textual features of the instructions generated above {instructions}, for solving the ask of solving problems? Make sure these features are independent of each other and not confounded. Give the answer directly without preparatory statements." ,
                },
            ],
            model=self.model,
        )
        features = response.choices[0].message.content.strip()
        return features

    def show_features_only(self, features: str) -> str:
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"Extract only features without any explanation: {features}, separating with commas.",
                }
            ],
            model=self.model,
        )
        show_features = response.choices[0].message.content.strip()
        return show_features

    def generate_counter_instructions(
        self,
        instructions: list[str],
        features: list[str],
    ) -> list[str]:
        counter_instructions = []

        for instruction in instructions:
            for feature in features:
                response = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": f"To let the large language model complete the task of solving problems, please generate a different LLM input instruction, similar to the following example: {self.base_instruction}. Give the answer directly without preparatory statements!",
                        },
                        {
                            "role": "assistant",
                            "content": str(instructions),
                        },
                        {
                            "role": "user",
                            "content": f"What are the measurable and improveable textual features of the instructions generated above {instructions}, for solving the ask of solving problems? Make sure these features are independent of each other and not confounded.  Give the answer directly without preparatory statements." ,
                        },
                        {
                            "role": "assistant",
                            "content": str(features),
                        },
                        {
                            "role": "user",
                            "content": f"To let the large language model complete the task of solving problems, based on the {instruction}, generate a counterfactual input instruction according the {feature}. When generating the counterfactual instruction, other features remain unchanged. Give the answer directly without any explanation.",
                        },
                    ],
                    model=self.model,
                )
                counter_instruction = response.choices[0].message.content.strip()
                counter_instructions.append(counter_instruction)

        return counter_instructions

    def label_instructions(
        self,
        baseIstruct,
        sample_instructs,
        features,
        instruction,
    ):
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"To let the large language model complete the task of solving problems, please generate a different LLM input instruction, similar to the following example: {baseIstruct}. Give the answer directly without preparatory statements!",
                },
                {
                    "role": "assistant",
                    "content": str(sample_instructs),
                },
                {
                    "role": "user",
                    "content": f"What are the measurable and improveable textual features of the instructions generated above {sample_instructs}, for solving the ask of solving problems? Make sure these features are independent of each other and not confounded.  Give the answer directly without preparatory statements." ,
                },
                {
                    "role": "assistant",
                    "content": str(features),
                },
                {
                    "role": "user",
                    "content": f"According to the order of the factors:{features}, score the instruction:{instruction} with 1 to 10. The final result must be a string of scores separated by commas. Give the answer directly without preparatory statements.",
                },
            ],
            model=self.model,
        )
        instruction_label = response.choices[0].message.content.strip()
        return instruction_label

    def clean_data(self, df: pd.DataFrame, show_features: str) -> pd.DataFrame:
        split_columns = df["instruction_label"].str.split(",", expand=True)

        feature_names = show_features.split(",")

        split_columns.columns = feature_names

        df = df.drop(columns=["instruction_label"])

        df = pd.concat([df, split_columns], axis=1)

        df.columns = df.columns.str.replace(" ", "")

        return df

    def run(self) -> pd.DataFrame:
        instructions = self.generate_instructions(num_instructions=10)

        df = pd.DataFrame()
        df["instruction"] = instructions

        features = self.extract_features(instructions)
        show_features = self.show_features_only(features)

        features = features.split("\n\n")

        counter_instructions = self.generate_counter_instructions(instructions, features)

        df_counter = pd.DataFrame()
        df_counter["instruction"] = counter_instructions

        df = pd.concat([df, df_counter])

        df = df.map(lambda x: x.strip('"') if isinstance(x, str) else x)

        instruction_labels = []
        for _, row in df.iterrows():
            instruction = row["instruction"]
            instruction_label = self.label_instructions(self.base_instruction, instructions, features, instruction)
            instruction_labels.append(instruction_label)
        df["instruction_label"] = instruction_labels


        df = self.clean_data(df, show_features)

        return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_instruction",
        type=str,
        default="Let's think step by step.",
        help="Base instruction for generating instructions",
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
        "--output",
        type=str,
        default="./instructions_data.csv",
        help="Path to the output file",
    )
    args = parser.parse_args()

    pipeline = DataPipeline(
        base_instruction=args.base_instruction,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
    )

    df = pipeline.run()

    df.to_csv(args.output, index=False, sep=",")


if __name__ == "__main__":
    main()
