from engines.gemini import generate_llm_output
from tasks.task import get_task

from verification.prompter.cot_prompter import cot_prompter
from verification.formatter.step_formatter import step_formatter
from verification.verify_and_revise.verify_and_revise import verify_and_revise
from verification.verifiers.step_co import StepCo
from verification.verifiers.general_purpose import GeneralPurposeVerifier

from extract_answer.extract_answer import extract_answer
from utils.result_to_json import format_reasoning_to_json


def process_verification(query, verifier, threshold=0.5, max_revisions=3):
    cot_prompt = cot_prompter(query)
    reasoning_path = generate_llm_output(cot_prompt)
    initial_steps = step_formatter(reasoning_path)

    print(f"Initial reasoning path with {len(initial_steps)} steps")
    for i, step in enumerate(initial_steps):
        print(f"Step {i+1}: {step[:100]}...")
    
    # Verify and revise each step
    final_steps = verify_and_revise(query, initial_steps, verifier, threshold, max_revisions)
    
    # Extract the final answer
    final_answer = extract_answer(final_steps, query)
    
    return {
        "initial_steps": initial_steps,
        "final_steps": final_steps,
        "answer": final_answer
    }

if __name__ == "__main__":
    verifier = GeneralPurposeVerifier()
    task = get_task("gpqa")
    print("TASK: ", task)
    result = process_verification(task, verifier)
    print(result)
    # format_reasoning_to_json(result, "exploration/results/general_purpose/mmlu_general_purpose_001.json")
    # format_reasoning_to_json(result, "exploration/results/step_co/bigbenchhard_step_co_result_001.json")