import re
from textgrad.variable import Variable
from typing import Union, List
from textgrad.engine import EngineLM
from textgrad.config import validate_engine_or_get_default
from .verifier import Verifier

class TextualVerifier(Verifier):
    """
    A verifier that uses an LLM to evaluate and improve reasoning steps.
    """
    
    def __init__(self, 
                verifier_engine: Union[str, EngineLM],
                step_eval_iterations: int = 3):
        """
        Initialize the TextualVerifier.
        
        Args:
            engine: The engine to use for evaluation
            eval_system_prompt: Custom evaluation prompt (optional)
            step_eval_iterations: Number of verification iterations per step
        """
        self.engine = validate_engine_or_get_default(verifier_engine)
        self.step_eval_iterations = step_eval_iterations
    
    def verify(self, instance: Variable, prompt: Variable, calculation: Variable) -> Variable:
        """
        Verify and improve the calculation through step-by-step analysis.
        
        Args:
            instance: The original problem/question
            prompt: The loss/optimizer prompt
            calculation: The result of prompt(instance) that needs verification
            
        Returns:
            Variable: Improved and verified calculation
        """
        print("🔍 Starting TextualVerifier...")
        # Step 1: Combine problem and solution for analysis
        full_context = f"Problem: {instance.value}\n\nSolution: {calculation.value}"
        
        # Step 2: Generate step-by-step reasoning
        reasoning_steps = self._generate_reasoning_steps(full_context)
        
        # Step 3: Verify each step
        verified_steps = self._verify_steps(instance.value, reasoning_steps)
        
        # Step 4: Create final improved solution
        improved_solution = self._create_final_solution(instance.value, verified_steps)
        
        print("✅ Verification complete!")
        return improved_solution
    
    def _generate_reasoning_steps(self, context: str) -> List[str]:
        """Generate step-by-step reasoning from the context."""
        
        cot_prompt = f"""
        Break down this solution into clear, logical steps. 
        Mark each step with <Step> and </Step> tags.
        
        {context}
        
        Let's think step by step:
        """
        
        reasoning_response = self.engine(cot_prompt)
        steps = self._extract_steps(reasoning_response)
        
        print(f"📋 Generated {len(steps)} reasoning steps")
        return steps
    
    def _extract_steps(self, reasoning_text: str) -> List[str]:
        """Extract steps from reasoning text using regex."""
        
        # Look for <Step>...</Step> patterns
        step_pattern = r"<Step>(.*?)</Step>"
        steps = re.findall(step_pattern, reasoning_text, re.DOTALL)
        
        # Clean up steps
        cleaned_steps = [step.strip() for step in steps if step.strip()]
        
        # If no tagged steps found, split by common patterns
        if not cleaned_steps:
            # Fallback: split by numbers or bullet points
            lines = reasoning_text.split('\n')
            cleaned_steps = [line.strip() for line in lines 
                           if line.strip() and len(line.strip()) > 10]
        
        return cleaned_steps[:5] if cleaned_steps else [reasoning_text]  # Limit to 5 steps
    
    def _verify_steps(self, question: str, steps: List[str]) -> List[str]:
        """Verify and improve each step."""
        
        verified_steps = []
        
        for i, step in enumerate(steps):
            print(f"🔧 Verifying step {i+1}/{len(steps)}")
            
            # Create verification prompt
            verification_prompt = f"""
            Question: {question}
            
            Previous steps: {' '.join(verified_steps)}
            
            Current step to verify: {step}
            
            Is this step correct? If not, provide a corrected version.
            Be concise and focus on accuracy.
            """
            
            # Get verification feedback
            verification_result = self.engine(verification_prompt)
            
            # For simplicity, use the verification result as the improved step
            verified_steps.append(verification_result)
        
        return verified_steps
    
    def _create_final_solution(self, question: str, verified_steps: List[str]) -> Variable:
        """Create the final improved solution from verified steps."""
        
        final_prompt = f"""
        Question: {question}
        
        Verified reasoning steps:
        {chr(10).join(f"{i+1}. {step}" for i, step in enumerate(verified_steps))}
        
        Based on these verified steps, provide a clear, final solution:
        """
        
        final_solution = self.engine(final_prompt)
        
        return Variable(
            final_solution,
            requires_grad=True,
            role_description="verified and improved solution"
        )