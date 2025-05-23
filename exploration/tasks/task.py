import random
from tasks.sample_datasets import get_sample_datasets

TASK_DESCRIPTION = {
    "bigbenchhard": "You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value.",
    "gpqa": "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.",
    "gsm8k": "You will answer a mathemetical reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value.",
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
            A) {sample["Correct Answer"]}
            B) {sample["Incorrect Answer 1"]}
            C) {sample["Incorrect Answer 2"]}
            D) {sample["Incorrect Answer 3"]}
            """
        case "gsm8k":
            task += sample["question"]

    return task