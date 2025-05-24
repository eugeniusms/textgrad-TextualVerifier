import re
from typing import List, Union, Dict
from textgrad.variable import Variable
from textgrad.engine import EngineLM
from textgrad.optimizer import TextualGradientDescent
from textgrad.verification.general_purpose_verifier import Verifier
from .optimizer_prompts import OPTIMIZER_SYSTEM_PROMPT

class VerifiedTextualGradientDescent(TextualGradientDescent):
    """
    A TextGrad optimizer that uses verification to improve reasoning steps.
    Inherits from TextualGradientDescent and adds verification capabilities.
    """
    
    def __init__(self, 
                 parameters: List[Variable],
                 verifier: Verifier,
                 threshold: float = 0.5,
                 max_revisions: int = 3,
                 verbose: int = 0,
                 engine: Union[EngineLM, str] = None,
                 constraints: List[str] = None,
                 new_variable_tags: List[str] = None,
                 optimizer_system_prompt: str = OPTIMIZER_SYSTEM_PROMPT,
                 in_context_examples: List[str] = None,
                 gradient_memory: int = 0):
        """
        Initialize the VerifiedTextualGradientDescent optimizer.
        
        Args:
            parameters: The parameters to optimize
            verifier: The verifier to use for evaluating reasoning steps
            threshold: Probability threshold for acceptable steps
            max_revisions: Maximum number of revision attempts per step
            verbose: Whether to print iterations
            engine: The engine to use for generating revisions
            constraints: List of constraints for optimization
            new_variable_tags: Tags for marking the new variable in the optimizer output
            optimizer_system_prompt: System prompt for the optimizer
            in_context_examples: In-context examples for optimization
            gradient_memory: Number of past gradients to store
        """
        # Initialize the parent TextualGradientDescent class
        super().__init__(
            parameters=parameters,
            verbose=verbose,
            engine=engine,
            constraints=constraints,
            new_variable_tags=new_variable_tags,
            optimizer_system_prompt=optimizer_system_prompt,
            in_context_examples=in_context_examples,
            gradient_memory=gradient_memory
        )
        
        # Add verification-specific attributes
        self.verifier = verifier
        self.threshold = threshold
        self.max_revisions = max_revisions

    def cot_prompter(self, query: str) -> str:
        """
        Create a Chain-of-Thought prompt for the given query.
        
        Args:
            query (str): The question/problem to solve
            
        Returns:
            str: The CoT prompt
        """
        prompt = f"""
        Mark the beginning and end of each reasoning step with <Step> 
        and </Step> tags. Let's think step by step.

        {query}
        """
        return prompt

    def step_formatter(self, reasoning_path: str) -> list:
        """
        Extract individual steps from a reasoning path.
        
        Args:
            reasoning_path (str): The reasoning path with step tags
            
        Returns:
            list: List of individual reasoning steps
        """
        # Use regex to extract content between <Step> and </Step> tags
        step_pattern = r"<Step>(.*?)</Step>"
        steps = re.findall(step_pattern, reasoning_path, re.DOTALL)
        
        # Clean up extracted steps
        return [step.strip() for step in steps]
    
    def verify_step(self, 
                    question: str, 
                    chain_so_far: List[str]) -> Dict:
        """
        Verify a single step in the reasoning chain.
        
        Args:
            question: The original problem
            chain_so_far: The reasoning chain up to and including the current step
            
        Returns:
            dict: Verification result with probability and revision decision
        """
        preceding_steps = "\n".join(chain_so_far)
        verifier_input = f"{question}\n{preceding_steps}"
        
        try:
            # Get probability that this step leads to correct answer
            probability = self.verifier.predict(verifier_input)
        except Exception as e:
            print(f"Error in verifier.predict(): {e}")
            probability = 0.5
        
        return {
            "probability": probability,
            "revise": probability < self.threshold
        }
    
    def revise_step(self, 
                    question: str, 
                    previous_steps: List[str], 
                    current_step: str, 
                    probability: float) -> str:
        """
        Revise a single step in the reasoning chain.
        
        Args:
            question: The original problem
            previous_steps: All verified steps before the current step
            current_step: The step to be revised
            probability: The probability score of the current step
            
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
        revised_output = self.engine(prompt)
        revised_steps = self.step_formatter(revised_output)
        
        # Return the revised step or the original if extraction failed
        if revised_steps and len(revised_steps) > 0:
            return revised_steps[0]
        else:
            return current_step
    
    def verify_and_revise(self, 
                          question: str, 
                          reasoning_chain: List[str]) -> List[str]:
        """
        Verify and revise each step in the reasoning chain until all steps meet the threshold.
        
        Args:
            question: The original problem
            reasoning_chain: List of reasoning steps
            
        Returns:
            list: The final verified and revised reasoning chain
        """
        verified_chain = []
        
        # Process each step in the reasoning chain
        for i, step in enumerate(reasoning_chain):
            if self.verbose > 0:
                print(f"\nVerifying step {i+1}:")
            
            current_step = step
            
            # Try to revise this step until it meets the threshold or max revisions reached
            for revision in range(self.max_revisions):
                current_chain = verified_chain + [current_step]
                verification = self.verify_step(question, current_chain)
                
                if self.verbose > 0:
                    print(f"Step {i+1} (Revision {revision}) probability: {verification['probability']:.4f}")
                
                # If step meets threshold, accept it and move to next step
                if verification['probability'] >= self.threshold:
                    if self.verbose > 0:
                        print(f"Step {i+1} meets threshold, moving to next step")
                    verified_chain.append(current_step)
                    break
                    
                # If step doesn't meet threshold and we haven't reached max revisions, revise it
                if revision < self.max_revisions - 1:
                    if self.verbose > 0:
                        print(f"Step {i+1} below threshold, revising...")
                    revised_step = self.revise_step(question, verified_chain, current_step, verification['probability'])
                    current_step = revised_step
                else:
                    # If we've reached max revisions, accept the current step and move on
                    if self.verbose > 0:
                        print(f"Warning: Step {i+1} still below threshold after {self.max_revisions} revisions. Accepting anyway.")
                    verified_chain.append(current_step)
        
        return verified_chain
    
    def step(self):
        """
        Perform a single optimization step using verification.
        
        This method first extracts reasoning steps from the parameter, verifies and revises them,
        and then passes the verified reasoning to the parent TextualGradientDescent.step() method
        for further optimization.
        """
        for parameter in self.parameters:
            # Extract reasoning steps using step tags
            reasoning_path = self.cot_prompter(parameter.value)
            initial_steps = self.step_formatter(reasoning_path)
            
            if not initial_steps:
                # If no steps found, treat the entire text as a single step
                initial_steps = [reasoning_path]
            
            # Extract the question from the parameter
            question = ""
            for pred in parameter.predecessors:
                if pred.get_role_description() == "question to the LLM" or "question" in pred.get_role_description().lower():
                    question = pred.value
                    break
            
            if self.verbose > 0:
                print(f"Initial reasoning path with {len(initial_steps)} steps")
                for i, step in enumerate(initial_steps):
                    print(f"Step {i+1}: {step[:100]}...")
            

            print("QUESTION", question)            
            # Verify and revise each step
            verified_steps = self.verify_and_revise(question, initial_steps)
            
            # Format the verified steps back into a cohesive reasoning
            verified_reasoning = "\n\n".join([f"<Step>{step}</Step>" for step in verified_steps])
            
            # Update the parameter with the verified reasoning
            parameter.set_value(verified_reasoning)
            
            if self.verbose > 0:
                print("-----------------------Verification Complete------------------------")
                print(parameter.value)
        
        # Apply the standard TextualGradientDescent optimization
        if self.verbose > 0:
            print("\nApplying standard optimization...")
        
        # Call the parent class's step method
        super().step()

    