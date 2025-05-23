import random
from tasks.sample_datasets import get_sample_datasets

TASK_DESCRIPTION = {
    "bigbenchhard": "You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value.",
    "gpqa": "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.",
    "gsm8k": "You will answer a mathemetical reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value.",
    "leetcode": "You will solve a hard coding problem from LeetCode. You will be given a prompt describing a problem. You need to write a function that passes all the tests.",
    "mmlu": "Given a multiple choice question, the goal is to select the correct final answer from the choices."
}

def get_task(dataset):
    sample_datasets = get_sample_datasets(dataset)
    randx = random.randint(0, len(sample_datasets)-1)
    sample = sample_datasets[randx]
    task = TASK_DESCRIPTION[dataset] + "\n\n"

    match dataset:
        case "bigbenchhard":
            task += sample["input"]
        case "gpqa":
            task += f"""
            {sample["Question"]}
            A) {sample["Correct Answer"]}
            B) {sample["Incorrect Answer 1"]}
            C) {sample["Incorrect Answer 2"]}
            D) {sample["Incorrect Answer 3"]}
            """
        case "gsm8k":
            task += sample["question"]
        case "leetcode":
            task += sample["content"]
        case "mmlu":
            task += f"""
            {sample["question"]}
            A) {sample["choices"][0]}
            B) {sample["choices"][1]}
            C) {sample["choices"][2]}
            D) {sample["choices"][3]}
            """

    return task