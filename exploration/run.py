import re
from engines.gemini import generate_llm_output
from prompter.cot_prompter import cot_prompter
from formatter.step_formatter import step_formatter
from verifiers.step_co import StepCo

QUERY = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they can be clearly resolved?

A) 10^-4 eV
B) 10^-11 eV
C) 10^-8 eV
D) 10^-9 eV
""".strip()

def process_verification(query, verifier, threshold=0.5, max_iterations=5):
    cot_prompt = cot_prompter(query)
    reasoning_path = generate_llm_output(cot_prompt)
    current_steps = step_formatter(reasoning_path)

    iterations = []

    for iteration in range(max_iterations):
        verification_result = verify_steps(query, current_steps, verifier, threshold)

        iterations.append({
            "iteration": iteration,
            "steps": current_steps.copy(),
            "verification": verification_result
        })

        if verification_result["all_correct"]:
            print(f"All steps correct after iteration {iteration}")
            break

        incorrect_step_index = verification_result["first_incorrect_step_index"]
        incorrect_step_prob = verification_result["step_probabilities"][incorrect_step_index]
        
        print(f"Iteration {iteration}: Found incorrect step at index {incorrect_step_index} with probability {incorrect_step_prob:.4f}")
        
        # Step 3.4: Revise incorrect steps
        current_steps = revise_steps(
            query, 
            current_steps, 
            incorrect_step_index, 
            incorrect_step_prob,
        )

    # Step 4: Extract the final answer
    final_answer = extract_answer(current_steps)
    
    return {
        "reasoning_steps": current_steps,
        "answer": final_answer,
        "iterations": iterations
    }

def verify_steps(question, steps, verifier, threshold):
    """
    Verify each step in the reasoning path using the process-supervised verifier.
    
    Args:
        question (str): The original problem
        steps (list): List of reasoning steps
        verifier: The process-supervised verifier model
        threshold (float): Probability threshold for determining incorrect steps
        
    Returns:
        dict: Verification results including step probabilities and first incorrect step
    """
    step_probabilities = []
    first_incorrect_step_index = -1
    
    # Check each step with the verifier
    for i, step in enumerate(steps):
        # Create input for the verifier by combining question and steps up to current step
        preceding_steps = "\n".join(steps[:i+1])
        verifier_input = f"{question}\n{preceding_steps}"
        
        # Get probability that this step leads to correct answer
        probability = verifier.predict(input_text=verifier_input)
        step_probabilities.append(probability)
        
        # Check if this is the first step below the threshold
        if probability < threshold and first_incorrect_step_index == -1:
            first_incorrect_step_index = i
    
    return {
        "step_probabilities": step_probabilities,
        "first_incorrect_step_index": first_incorrect_step_index,
        "all_correct": first_incorrect_step_index == -1
    }

def revise_steps(question, steps, incorrect_step_index, incorrect_step_prob, llm):
    """
    Revise the incorrect steps in the reasoning path.
    
    Args:
        question (str): The original problem
        steps (list): List of reasoning steps
        incorrect_step_index (int): Index of the first incorrect step
        incorrect_step_prob (float): Probability of the incorrect step
        llm: The language model to use for revision
        
    Returns:
        list: Revised list of reasoning steps
    """
    # Keep the correct steps (before the incorrect step)
    correct_steps = steps[:incorrect_step_index]
    
    # Format all steps with tags for the prompt
    all_steps_formatted = "\n".join([f"<Step>{step}</Step>" for step in steps])
    
    # Create prompt for revising the incorrect steps
    prompt = f"""Q: {question}. A: {all_steps_formatted}. 
    The probability that step <Step>{steps[incorrect_step_index]}</Step> leads to the correct answer is {incorrect_step_prob:.4f}. 
    Please revise steps {incorrect_step_index + 1} to {len(steps)} while keeping steps 1 to {incorrect_step_index} unchanged to increase the probability that the revised steps lead to the correct answer."""
    
    # Generate revised steps
    revised_output = generate_llm_output(prompt)
    
    # Extract steps from the revised output
    revised_steps = step_formatter(revised_output)
    
    # Combine correct steps with revised steps
    if incorrect_step_index < len(revised_steps):
        return correct_steps + revised_steps[incorrect_step_index:]
    else:
        # Handle case where revised output has fewer steps
        return correct_steps + revised_steps

def extract_answer(steps):
    """
    Extract the final answer from the reasoning path.
    
    Args:
        steps (list): List of reasoning steps
        
    Returns:
        str: The final answer
    """
    # In most cases, the final answer is in the last step
    last_step = steps[-1]
    
    # Simple extraction - look for numeric answer or final statement
    # In practice, you might need more sophisticated parsing
    
    # Try to find a numeric answer
    number_match = re.search(r'(\d+(\.\d+)?)', last_step)
    if number_match:
        return number_match.group(1)
    
    # If no clear numeric answer, return the last step
    return last_step

if __name__ == "__main__":
    verifier = StepCo()
    result = process_verification(QUERY, verifier)
    print(result)