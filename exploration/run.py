from engines.gemini import generate_llm_output
from prompter.cot_prompter import cot_prompter
from formatter.step_formatter import step_formatter
from verify_and_revise.verify_and_revise import verify_and_revise
from extract_answer.extract_answer import extract_answer
from verifiers.step_co import StepCo
from verifiers.general_purpose import GeneralPurposeVerifier
from utils.result_to_json import format_reasoning_to_json

QUERY = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they can be clearly resolved?

A) 10^-4 eV
B) 10^-11 eV
C) 10^-8 eV
D) 10^-9 eV
""".strip()

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
    verifier = StepCo()
    result = process_verification(QUERY, verifier)
    format_reasoning_to_json(result, "exploration/results/step_co/step_co_result_002.json")