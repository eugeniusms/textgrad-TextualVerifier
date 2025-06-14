import re
from textgrad.variable import Variable
from typing import Union, List
from textgrad.engine import EngineLM
from textgrad.config import validate_engine_or_get_default
from .verifier import Verifier
from .verifier_prompts_experiment import (
    VARIANT_GENERATION_PROMPT_WITH_CONTEXT,
    VOTING_PROMPT_WITH_CONTEXT,
    DECISION_PROMPT,
    get_final_step_instruction,
    get_voting_final_instruction
)

class TextualVerifierExperiment:
    """
    Improved verifier with better step consistency and format control.
    """

    def __init__(self, verifier_engine: Union[str, EngineLM], step_eval_iterations: int = 3, logger: bool = False):
        self.engine = validate_engine_or_get_default(verifier_engine)
        self.step_eval_iterations = step_eval_iterations
        self.logger = logger
        
    def verify(self, instance: Variable, prompt: Variable, calculation: Variable) -> Variable:
        """Main verification function with improved step handling."""
        if self.logger:
            print("INFO: Starting improved textual verification...")

        # Extract and clean steps
        reasoning_steps = self._extract_steps_from_response(calculation.value)
        
        if not reasoning_steps:
            if self.logger:
                print("WARN: No steps found, returning original")
            return calculation

        if self.logger:
            print(f"INFO: Found {len(reasoning_steps)} steps to verify")

        # Verify each step with better context management
        verified_steps = self._verify_steps_with_improved_context(
            instance.value, prompt.value, reasoning_steps
        )
        
        # Create final solution
        final_solution = self._create_final_solution(verified_steps)
        
        # Decide whether to use improved version
        decision = self._make_improved_decision(reasoning_steps, verified_steps)
        
        if decision == "REPLACE":
            result = self._format_as_tagged_steps(final_solution)
        else:
            result = calculation.value
            
        if self.logger:
            print(f"INFO: Verification complete, decision: {decision}")

        return Variable(result, requires_grad=True, role_description="verified calculation")

    def _extract_steps_from_response(self, response: str) -> List[str]:
        """Extract steps with better cleaning."""
        # Try tagged format first
        step_pattern = r"<Step>(.*?)</Step>"
        steps = re.findall(step_pattern, response, re.DOTALL)
        
        if steps:
            cleaned_steps = []
            for step in steps:
                # Clean up the step content
                cleaned = step.strip()
                # Remove any "Step X:" prefixes that might be inside
                cleaned = re.sub(r'^Step \d+:\s*', '', cleaned)
                if cleaned:
                    cleaned_steps.append(cleaned)
            return cleaned_steps
        
        # Fallback to line-based extraction
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        steps = []
        current_step = ""
        
        for line in lines:
            if line.startswith("Step ") and ":" in line:
                if current_step:
                    steps.append(current_step.strip())
                current_step = line.split(":", 1)[1].strip()
            elif current_step:
                current_step += " " + line
            elif len(line) > 10:  # Potential step without prefix
                current_step = line
                
        if current_step:
            steps.append(current_step.strip())
            
        return steps

    def _verify_steps_with_improved_context(self, instance: str, prompt: str, steps: List[str]) -> List[str]:
        """Verify steps with improved context management."""
        verified_steps = []
        
        for i, step in enumerate(steps):
            if self.logger:
                print(f"INFO: Verifying step {i+1}/{len(steps)}")
            
            # Build clean context from verified steps
            context = self._build_context(verified_steps)
            is_final = (i == len(steps) - 1)
            
            # Generate improved variants
            variants = self._generate_improved_variants(
                instance, prompt, step, context, is_final
            )
            
            # Select best variant
            best_step = self._select_best_variant(
                step, variants, context, i, len(steps)
            )
            
            # Clean and validate the selected step
            cleaned_step = self._clean_step_output(best_step, is_final)
            verified_steps.append(cleaned_step)
        
        return verified_steps

    def _build_context(self, verified_steps: List[str]) -> str:
        """Build clean context from verified steps."""
        if not verified_steps:
            return "None (this is the first step)"
        
        context_lines = []
        for i, step in enumerate(verified_steps):
            # Remove any answer lines from context
            step_content = re.sub(r'Answer:.*$', '', step, flags=re.MULTILINE).strip()
            if step_content:
                context_lines.append(f"Step {i+1}: {step_content}")
        
        return "\n".join(context_lines)

    def _generate_improved_variants(self, instance: str, prompt: str, step: str, context: str, is_final: bool) -> List[str]:
        """Generate variants with improved prompting."""
        variants = []
        final_instruction = get_final_step_instruction(is_final)
        
        for i in range(self.step_eval_iterations):
            variant_prompt = VARIANT_GENERATION_PROMPT_WITH_CONTEXT.format(
                problem=instance,
                approach=prompt, 
                context=context,
                current_step=step,
                is_final=is_final,
                final_instruction=final_instruction
            )
            
            try:
                variant = self.engine(variant_prompt).strip()
                if variant:
                    variants.append(variant)
            except Exception as e:
                if self.logger:
                    print(f"WARN: Failed to generate variant {i+1}: {e}")
        
        return variants if variants else [step]  # Fallback to original

    def _select_best_variant(self, original: str, variants: List[str], context: str, step_index: int, total_steps: int) -> str:
        """Select best variant with improved voting."""
        if len(variants) <= 1:
            return variants[0] if variants else original
        
        # Format variants for voting
        formatted_variants = []
        for i, variant in enumerate(variants):
            formatted_variants.append(f"Candidate {i+1}: {variant}")
        
        variants_text = "\n\n".join(formatted_variants)
        voting_instruction = get_voting_final_instruction(step_index, total_steps)
        
        voting_prompt = VOTING_PROMPT_WITH_CONTEXT.format(
            context=context,
            original_step=original,
            candidates=variants_text,
            voting_instruction=voting_instruction
        )
        
        try:
            selection = self.engine(voting_prompt).strip()
            # Clean the selection to remove any "Candidate X:" prefixes
            cleaned_selection = re.sub(r'^Candidate \d+:\s*', '', selection)
            return cleaned_selection if cleaned_selection else variants[0]
        except Exception as e:
            if self.logger:
                print(f"WARN: Voting failed: {e}")
            return variants[0]

    def _clean_step_output(self, step: str, is_final: bool) -> str:
        """Clean step output to ensure consistent formatting."""
        # Remove any step prefixes
        cleaned = re.sub(r'^Step \d+:\s*', '', step)
        
        # Ensure proper answer format for final step
        if is_final and "Answer" in cleaned:
            # Standardize answer format
            cleaned = re.sub(r'Answer\s*:?\s*([^\n]+)', r'Answer: \1', cleaned)
        
        return cleaned.strip()

    def _create_final_solution(self, verified_steps: List[str]) -> str:
        """Create final solution from verified steps."""
        return "\n".join(f"Step {i+1}: {step}" for i, step in enumerate(verified_steps))

    def _make_improved_decision(self, original_steps: List[str], verified_steps: List[str]) -> str:
        """Make decision with improved comparison."""
        original_text = "\n".join(f"Step {i+1}: {step}" for i, step in enumerate(original_steps))
        verified_text = "\n".join(f"Step {i+1}: {step}" for i, step in enumerate(verified_steps))
        
        decision_prompt = DECISION_PROMPT.format(
            original_steps=original_text,
            verified_steps=verified_text
        )
        
        try:
            response = self.engine(decision_prompt).strip()
            if "REPLACE" in response.upper():
                return "REPLACE"
            else:
                return "SUFFICIENT"
        except Exception as e:
            if self.logger:
                print(f"WARN: Decision making failed: {e}")
            return "SUFFICIENT"  # Conservative fallback

    def _format_as_tagged_steps(self, solution: str) -> str:
        """Format solution with proper step tags."""
        lines = solution.split('\n')
        formatted_lines = []
        
        for line in lines:
            if line.startswith('Step '):
                # Extract step content
                step_content = line.split(':', 1)[1].strip() if ':' in line else line
                formatted_lines.append(f"<Step>{step_content}</Step>")
            elif line.strip():
                # Handle continuation lines
                if formatted_lines:
                    # Add to previous step
                    prev_step = formatted_lines[-1]
                    if prev_step.endswith('</Step>'):
                        content = prev_step[6:-7]  # Remove <Step> and </Step>
                        formatted_lines[-1] = f"<Step>{content}\n{line}</Step>"
                    else:
                        formatted_lines.append(line)
                else:
                    formatted_lines.append(f"<Step>{line}</Step>")
        
        return '\n'.join(formatted_lines)
