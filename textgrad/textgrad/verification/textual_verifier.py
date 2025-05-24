from typing import List, Union
import re
from textgrad.engine import EngineLM
from textgrad.autograd import LLMCall
from textgrad.variable import Variable
from textgrad.config import validate_engine_or_get_default
from textgrad.verification.verifier import Verifier

class TextualVerifier(Verifier):
    """
    A verifier that uses an LLM to evaluate reasoning steps and 
    converts the textual evaluation to a probability score.
    """
    
    def __init__(self, 
                engine: Union[str, EngineLM],
                eval_system_prompt: Variable = None,
                step_eval_iterations: int = 3):
        """
        Initialize the TextualVerifier verifier.
        
        Args:
            engine: The engine to use for evaluation
        """
        self.engine = validate_engine_or_get_default(engine)
        self.eval_system_prompt = eval_system_prompt
        self.step_eval_iterations = step_eval_iterations
    
    def verify(self, instance: Variable, calculation: Variable) -> str:
        """
        Verify the calculation of instance.
        
        Args:
            instance: The variable to evaluate (ex: solution/prompt)
            calculation: Result calculated by loss/optimizer function
            
        Returns:
            str: Verification result (updated calculation)
        """
        print("Verifier: Textual")
        # Step 1: Extract steps from reasoning (using your step_formatter logic)
        question = instance + calculation
        cot_prompt = self.cot_prompter(question)
        reasoning_path = self.engine(cot_prompt)
        initial_steps = self._step_formatter(reasoning_path)
        
        if not initial_steps:
            # If no steps found, treat the whole text as one step
            initial_steps = [reasoning_path]
        
        print(f"Initial reasoning path with {len(initial_steps)} steps")
        for i, step in enumerate(initial_steps):
            print(f"Step {i+1}: {step[:100]}...")
        
        # Step 2: Verify and revise each step (using your verify_loss logic)
        final_steps = self._verify_loss(question.value, initial_steps)
        
        # Step 3: Create verified reasoning text
        verified_reasoning_text = "\n".join([f"<Step>{step}</Step>" for step in final_steps])
        
        # Step 4: Final evaluation using the eval_system_prompt
        evaluation_input = f"Question: {question.value}\n\nVerified Reasoning:\n{verified_reasoning_text}"
        
        # Create a formatted call for evaluation
        final_evaluation = self.engine(f"{self.eval_system_prompt.value}\n\n{evaluation_input}")

        print("FINAL EVALUATION: ", final_evaluation)
        
        return Variable(final_evaluation, requires_grad=True, 
                       role_description="evaluation of verified reasoning")
    
    def cot_prompter(self, query):
        initial_reasoning_path = f"""
            Mark the beginning and end of each reasoning step with <Step> 
            and </Step> tags. Q: q. A: Let's think step by step.

            {query}
        """
        return initial_reasoning_path
    
    def _step_formatter(self, reasoning_path: str) -> List[str]:
        """
        Extract individual steps from a reasoning path.
        This implements your step_formatter function.
        """
        # Use regex to extract content between <Step> and </Step> tags
        step_pattern = r"<Step>(.*?)</Step>"
        steps = re.findall(step_pattern, reasoning_path, re.DOTALL)
        
        # Clean up extracted steps
        return [step.strip() for step in steps]

    def _find_step_loss(self, question: str, chain_so_far: List[str]) -> str:
        """
        Find a single step loss in the reasoning chain.
        This implements your find_step_loss function.
        """
        preceding_steps = "\n".join(chain_so_far)
        newest_step = chain_so_far[-1]
        
        loss_order = Variable("""Evaluate whether this step is correct given the question and previous steps. 
                                    Consider mathematical accuracy, logical consistency, and relevance to solving the question.
                                    Create a textual feedback that represents error and correction for evaluated step. 
                                    Please concise one sentence.
        """, requires_grad=False, role_description="system prompt for the evaluation")
        step = Variable(f"""
                QUESTION:
                {question}

                PREVIOUS STEPS:
                {preceding_steps}

                STEP TO EVALUATE:
                {newest_step}                
                """, requires_grad=False, role_description="step")
        loss = LLMCall(self.engine, loss_order)
        result = loss(step)

        return result

    def _verify_loss(self, question: str, reasoning_chain: List[str]) -> List[str]:
        """
        Verify and revise each step in the reasoning chain.
        This implements your verify_loss function.
        """
        verified_chain = []
        
        # Process each step in the reasoning chain
        for i, step in enumerate(reasoning_chain):
            print(f"\nVerifying step {i+1}:")
            
            current_step = step
            
            # Running to prepare consensus/voting
            for step_eval in range(self.step_eval_iterations):
                current_chain = verified_chain + [current_step]
                loss = self._find_step_loss(question, current_chain)
                verified_chain.append(current_step)

                print(f"Step {i+1} (Eval {step_eval}) Loss: {loss}")
        
        return verified_chain