import ast
import time
from textgrad.variable import Variable
from typing import Union, List
from textgrad.engine import EngineLM
from textgrad.config import validate_engine_or_get_default

# Create an enhanced version of TextualVerifierExperiment with tracking
class TextualVerifierExperiment:
    """Enhanced TextualVerifierExperiment with comprehensive tracking"""
    
    def __init__(self, verifier_engine: Union[str, EngineLM], step_eval_iterations: int = 3, 
                 logger: bool = False, tracker=None):
        self.engine = validate_engine_or_get_default(verifier_engine)
        self.step_eval_iterations = step_eval_iterations
        self.logger = logger
        self.tracker = tracker  # Add tracker
        
    def verify(self, instance: Variable, prompt: Variable, calculation: Variable) -> Variable:
        """Main verification function with comprehensive tracking"""
        if self.logger:
            print("INFO: Starting tracked textual verification...")

        # Extract and clean steps
        reasoning_steps = self._extract_steps_from_response(calculation.value)
        
        if not reasoning_steps:
            if self.logger:
                print("WARN: No steps found, returning original")
            return calculation

        if self.logger:
            print(f"INFO: Found {len(reasoning_steps)} steps to verify")

        # Verify each step with comprehensive tracking
        verified_steps = self._verify_steps_with_tracking(
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

    def track_detailed_step_verification(self, step_index: int, original_step: str, 
                                       variants: List[str], selected_variant: str,
                                       selection_reason: str, processing_time_ms: float,
                                       llm_calls: List):
        """Override to use enhanced tracking"""
        if self.tracker and hasattr(self.tracker, 'track_detailed_step_verification'):
            self.tracker.track_detailed_step_verification(
                step_index, original_step, variants, selected_variant,
                selection_reason, processing_time_ms, llm_calls
            )
        else:
            # Fallback to regular tracking
            if self.tracker:
                self.tracker.track_step_verification(
                    step_index, original_step, variants, selected_variant,
                    selection_reason, processing_time_ms, llm_calls
                )

    def _verify_steps_with_tracking(self, instance: str, prompt: str, steps: List[str]) -> List[str]:
        """Override to capture detailed information"""
        verified_steps = []
        
        for i, step in enumerate(steps):
            step_start_time = time.time()
            step_llm_calls = []
            
            if self.logger:
                print(f"INFO: Verifying step {i+1}/{len(steps)}")
            
            # Build clean context from verified steps
            context = self._build_context(verified_steps)
            is_final = (i == len(steps) - 1)
            
            # Generate improved variants with tracking
            variants, variant_calls = self._generate_improved_variants_with_tracking(
                instance, prompt, step, context, is_final, i
            )
            step_llm_calls.extend(variant_calls)
            
            # Select best variant with tracking
            best_step, selection_calls = self._select_best_variant_with_tracking(
                step, variants, context, i, len(steps)
            )
            step_llm_calls.extend(selection_calls)
            
            # Clean and validate the selected step
            cleaned_step = self._clean_step_output(best_step, is_final)
            verified_steps.append(cleaned_step)
            
            # Track step-level metrics with detailed information
            step_processing_time = (time.time() - step_start_time) * 1000
            self.track_detailed_step_verification(
                step_index=i,
                original_step=step,
                variants=variants,
                selected_variant=cleaned_step,
                selection_reason="voting_based_selection",
                processing_time_ms=step_processing_time,
                llm_calls=step_llm_calls
            )
        
        return verified_steps

    def _generate_improved_variants_with_tracking(self, instance: str, prompt: str, step: str, 
                                                 context: str, is_final: bool, step_index: int):
        """Generate diverse variants with better prompting"""
        variants = []
        llm_calls = []
        
        from textgrad.verifier.verifier_prompts_experiment import (
            VARIANT_GENERATION_PROMPT_WITH_CONTEXT,
            get_final_step_instruction
        )
        
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
                # Track the LLM call with randomness for diversity
                call_start_time = time.time()
                variant = self.engine(
                    variant_prompt,
                    temperature=0.8,  # High temperature for diversity
                    top_p=0.9         # Nucleus sampling
                ).strip()
                call_latency = (time.time() - call_start_time) * 1000

                variants.append(variant)
                
               # Track this LLM call
                if self.tracker:
                    llm_call = self.tracker.track_llm_call(
                        stage="variant_generation",
                        step_index=step_index,
                        prompt=variant_prompt,
                        response=variant,
                        latency_ms=call_latency,
                        success=True
                    )
                    llm_calls.append(llm_call)
                
            except Exception as e:
                # Track failed LLM call
                if self.tracker:
                    llm_call = self.tracker.track_llm_call(
                        stage="variant_generation",
                        step_index=step_index,
                        prompt=variant_prompt,
                        response="",
                        latency_ms=0,
                        success=False,
                        error=str(e)
                    )
                    llm_calls.append(llm_call)
                
                if self.logger:
                    print(f"WARN: Failed to generate variant {i+1}: {e}")
        
        return variants if variants else [step], llm_calls

    def _select_best_variant_with_tracking(self, original: str, variants: List[str], 
                                          context: str, step_index: int, total_steps: int):
        """Select best variant with LLM call tracking"""
        llm_calls = []
        
        if len(variants) <= 1:
            return variants[0] if variants else original, llm_calls
        
        from textgrad.verifier.verifier_prompts_experiment import (
            VOTING_PROMPT_WITH_CONTEXT,
            get_voting_final_instruction
        )
        
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
            # Track the voting LLM call
            call_start_time = time.time()
            selection = self.engine(voting_prompt).strip()
            call_latency = (time.time() - call_start_time) * 1000
            
            # Track this LLM call
            if self.tracker:
                llm_call = self.tracker.track_llm_call(
                    stage="voting",
                    step_index=step_index,
                    prompt=voting_prompt,
                    response=selection,
                    latency_ms=call_latency,
                    success=True
                )
                llm_calls.append(llm_call)
            
            # Clean the selection to remove any "Candidate X:" prefixes
            import re
            cleaned_selection = re.sub(r'^Candidate \d+:\s*', '', selection)
            return cleaned_selection if cleaned_selection else variants[0], llm_calls
            
        except Exception as e:
            # Track failed voting call
            if self.tracker:
                llm_call = self.tracker.track_llm_call(
                    stage="voting",
                    step_index=step_index,
                    prompt=voting_prompt,
                    response="",
                    latency_ms=0,
                    success=False,
                    error=str(e)
                )
                llm_calls.append(llm_call)
            
            if self.logger:
                print(f"WARN: Voting failed: {e}")
            return variants[0], llm_calls

    def _make_improved_decision(self, original_steps: List[str], verified_steps: List[str]) -> str:
        """Make decision with LLM call tracking"""
        from textgrad.verifier.verifier_prompts_experiment import DECISION_PROMPT
        
        original_text = "\n".join(f"Step {i+1}: {step}" for i, step in enumerate(original_steps))
        verified_text = "\n".join(f"Step {i+1}: {step}" for i, step in enumerate(verified_steps))
        
        decision_prompt = DECISION_PROMPT.format(
            original_steps=original_text,
            verified_steps=verified_text
        )
        
        try:
            # Track the decision LLM call
            call_start_time = time.time()
            response = self.engine(decision_prompt).strip()
            call_latency = (time.time() - call_start_time) * 1000
            
            # Track this LLM call
            if self.tracker:
                self.tracker.track_llm_call(
                    stage="decision",
                    step_index=-1,  # Final decision, not step-specific
                    prompt=decision_prompt,
                    response=response,
                    latency_ms=call_latency,
                    success=True
                )
            
            if "REPLACE" in response.upper():
                return "REPLACE"
            else:
                return "SUFFICIENT"
                
        except Exception as e:
            # Track failed decision call
            if self.tracker:
                self.tracker.track_llm_call(
                    stage="decision",
                    step_index=-1,
                    prompt=decision_prompt,
                    response="",
                    latency_ms=0,
                    success=False,
                    error=str(e)
                )
            
            if self.logger:
                print(f"WARN: Decision making failed: {e}")
            return "SUFFICIENT"  # Conservative fallback

    # Copy all other methods from original TextualVerifierExperiment
    def _extract_steps_from_response(self, response: str) -> List[str]:
        """Extract steps with better cleaning - copy from original"""
        import re
        # Try tagged format first
        step_pattern = r"<Step>(.*?)</Step>"
        steps = re.findall(step_pattern, response, re.DOTALL)
        
        if steps:
            cleaned_steps = []
            for step in steps:
                cleaned = step.strip()
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
            elif len(line) > 10:
                current_step = line
                
        if current_step:
            steps.append(current_step.strip())
            
        return steps

    def _build_context(self, verified_steps: List[str]) -> str:
        """Build clean context from verified steps - copy from original"""
        if not verified_steps:
            return "None (this is the first step)"
        
        context_lines = []
        import re
        for i, step in enumerate(verified_steps):
            step_content = re.sub(r'Answer:.*$', '', step, flags=re.MULTILINE).strip()
            if step_content:
                context_lines.append(f"Step {i+1}: {step_content}")
        
        return "\n".join(context_lines)

    def _clean_step_output(self, step: str, is_final: bool) -> str:
        """Clean step output - copy from original"""
        import re
        cleaned = re.sub(r'^Step \d+:\s*', '', step)
        
        if is_final and "Answer" in cleaned:
            cleaned = re.sub(r'Answer\s*:?\s*([^\n]+)', r'Answer: \1', cleaned)
        
        return cleaned.strip()

    def _create_final_solution(self, verified_steps: List[str]) -> str:
        """Create final solution - copy from original"""
        return "\n".join(f"Step {i+1}: {step}" for i, step in enumerate(verified_steps))

    def _format_as_tagged_steps(self, solution: str) -> str:
        """Format solution with proper step tags - copy from original"""
        lines = solution.split('\n')
        formatted_lines = []
        
        for line in lines:
            if line.startswith('Step '):
                step_content = line.split(':', 1)[1].strip() if ':' in line else line
                formatted_lines.append(f"<Step>{step_content}</Step>")
            elif line.strip():
                if formatted_lines:
                    prev_step = formatted_lines[-1]
                    if prev_step.endswith('</Step>'):
                        content = prev_step[6:-7]
                        formatted_lines[-1] = f"<Step>{content}\n{line}</Step>"
                    else:
                        formatted_lines.append(line)
                else:
                    formatted_lines.append(f"<Step>{line}</Step>")
        
        return '\n'.join(formatted_lines)