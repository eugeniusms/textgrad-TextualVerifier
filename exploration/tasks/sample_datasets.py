# https://huggingface.co/datasets/maveriq/bigbenchhard
bigbenchhard = [
    {
        "input": "not not False and not not not False is",
        "target": "False"
    }
]

# https://huggingface.co/datasets/Idavidrein/gpqa
gpqa = [
    {
        "Question": "Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they can be clearly resolved?",
        "Incorrect Answer 1": "10^-11 eV",
        "Incorrect Answer 2": "10^-8 eV",
        "Incorrect Answer 3": "10^-9 eV",
        "Correct Answer": "10^-4 eV"
    }
]

# https://huggingface.co/datasets/openai/gsm8k
gsm8k = [
    {
        "question": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
        "answer": "He writes each friend 3*2=<<3*2=6>>6 pages a week\nSo he writes 6*2=<<6*2=12>>12 pages every week\nThat means he writes 12*52=<<12*52=624>>624 pages a year\n#### 624"
    }
]

def get_sample_datasets(dataset):
    if dataset == "bigbenchhard":
        return bigbenchhard
    if dataset == "gpqa":
        return gpqa
    if dataset == "gsm8k":
        return gsm8k
    return [{}]