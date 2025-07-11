from engines.gemini import generate_llm_output
from verification.formatter.step_formatter import step_formatter

def verify_and_revise(question, 
                    reasoning_chain, 
                    verifier, 
                    threshold=0.5, 
                    max_revisions=3):
    """
    Verify and revise each step in the reasoning chain until all steps meet the threshold.
    
    Args:
        question (str): The original problem
        reasoning_chain (list): List of reasoning steps
        verifier: The verifier model
        threshold (float): Probability threshold for acceptable steps
        max_revisions (int): Maximum number of revision attempts per step
        
    Returns:
        list: The final verified and revised reasoning chain
    """
    verified_chain = []
    
    # Process each step in the reasoning chain
    for i, step in enumerate(reasoning_chain):
        print(f"\nVerifying step {i+1}:")
        
        current_step = step
        
        # Try to revise this step until it meets the threshold or max revisions reached
        for revision in range(max_revisions):
            current_chain = verified_chain + [current_step]
            verification = verify_step(question, current_chain, verifier)
            
            print(f"Current chain: {current_chain}")
            print(f"Step {i+1} (Revision {revision}) probability: {verification['probability']:.4f}")
            
            # If step meets threshold, accept it and move to next step
            if verification['probability'] >= threshold:
                print(f"Step {i+1} meets threshold, moving to next step")
                verified_chain.append(current_step)
                break
                
            # If step doesn't meet threshold and we haven't reached max revisions, revise it
            if revision < max_revisions - 1:
                print(f"Step {i+1} below threshold, revising...")
                revised_step = revise_step(question, verified_chain, current_step, verification['probability'])
                current_step = revised_step
            else:
                # If we've reached max revisions, accept the current step and move on
                print(f"Warning: Step {i+1} still below threshold after {max_revisions} revisions. Accepting anyway.")
                verified_chain.append(current_step)
    
    return verified_chain

def verify_step(question, 
                chain_so_far, 
                verifier):
    """
    Verify a single step in the reasoning chain.
    
    Args:
        question (str): The original problem
        chain_so_far (list): The reasoning chain up to and including the current step
        verifier: The verifier model
        
    Returns:
        dict: Verification result with probability and revision decision
    """
    preceding_steps = "\n".join(chain_so_far)
    verifier_input = f"{question}\n{preceding_steps}"
    
    try:
        # Get probability that this step leads to correct answer
        probability = verifier.predict(input_text=verifier_input)
    except Exception as e:
        print(f"Error in verifier.predict(): {e}")
        probability = 0.5
    
    return {
        "probability": probability,
        "revise": probability < 0.5
    }

def revise_step(question, 
                previous_steps, 
                current_step, 
                probability):
    """
    Revise a single step in the reasoning chain.
    
    Args:
        question (str): The original problem
        previous_steps (list): All verified steps before the current step
        current_step (str): The step to be revised
        probability (float): The probability score of the current step
        
    Returns:
        str: The revised step
    """
    # Format previous steps with tags
    previous_formatted = ""
    if previous_steps:
        previous_formatted = "\n".join([f"<Step>{step}</Step>" for step in previous_steps])
        previous_formatted = f"Previous steps:\n{previous_formatted}\n\n"
    
    # Create prompt for revising the step
    prompt = f"""Q: {question}
    
    {previous_formatted}Current step to revise:
    <Step>{current_step}</Step>

    The probability that this step leads to the correct answer is {probability:.4f}, which is below the acceptable threshold.
    Please revise ONLY this step to increase the probability that it leads to the correct answer.
    Your revision should be more accurate, clear, and logical. Make sure it follows directly from the previous steps.

    Provide your revised step between <Step> and </Step> tags."""
    
    # Generate & extract the revised step
    revised_output = generate_llm_output(prompt)
    revised_steps = step_formatter(revised_output)
    
    # Return the revised step or the original if extraction failed
    if revised_steps and len(revised_steps) > 0:
        return revised_steps[0]
    else:
        return current_step