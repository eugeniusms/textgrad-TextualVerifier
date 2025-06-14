"""
Textual Verification System - Refactored for clarity and maintainability.

This module provides a generic verification system that can verify calculations
by breaking them down into steps, generating variants, and using majority voting.
"""

import re
from typing import Union, List
from textgrad.variable import Variable
from textgrad.engine import EngineLM
from textgrad.config import validate_engine_or_get_default
from .verifier import Verifier
from .verifier_prompts_generic import (
    DEFAULT_VERIFICATION_TASK_PROMPTS,
    COT_PROMPT,
    VARIANT_GENERATION_PROMPT,
    MAJORITY_VOTING_PROMPT,
)


class TextualVerifierGeneric(Verifier):
    """
    A generic textual verifier that validates calculations through:
    1. Breaking down calculations into steps (optional)
    2. Generating multiple verification variants
    3. Using majority voting to select the best result
    
    The verification process follows this pattern:
    instance + instruction => calculation
    instance + calculation + verification_prompt => verified_calculation
    """
    
    def __init__(
        self,
        verifier_engine: Union[str, EngineLM],
        use_cot_generation: bool = False,
        use_step_breakdown: bool = True,
        verification_task_prompts: List[str] = DEFAULT_VERIFICATION_TASK_PROMPTS,
        enable_logging: bool = False
    ):
        """
        Initialize the verifier.
        
        Args:
            verifier_engine: LLM engine for verification
            use_cot_generation: Whether to use Chain-of-Thought generation
            use_step_breakdown: Whether to break calculations into steps
            verification_task_prompts: List of prompts for verification tasks
            enable_logging: Whether to enable debug logging
        """
        self.engine = validate_engine_or_get_default(verifier_engine)
        self.use_cot_generation = use_cot_generation
        self.use_step_breakdown = use_step_breakdown
        self.verification_task_prompts = verification_task_prompts
        self.enable_logging = enable_logging
        
    def verify(
        self, 
        instance: Variable, 
        instruction: Variable, 
        calculation: Variable
    ) -> Variable:
        """
        Verify a calculation through step-by-step validation.
        
        Args:
            instance: The input data
            instruction: The instruction applied to the instance
            calculation: The output/result of applying instruction to instance
            
        Returns:
            Variable: Verified calculation wrapped in <VERIFIED> tags
        """
        # Step 1: Optionally generate Chain-of-Thought breakdown
        processed_calculation = self._process_calculation(calculation.value)
        
        # Step 2: Break into steps for verification
        verification_steps = self._get_verification_steps(processed_calculation)
        
        # Step 3: Verify each step through variant generation and voting
        verified_steps = self._verify_steps(
            instance.value, 
            instruction.value, 
            verification_steps
        )
        
        # Step 4: Format final result
        final_result = self._format_verified_result(verified_steps)
        
        return Variable(
            final_result, 
            requires_grad=True, 
            role_description="verified calculation"
        )

    def _process_calculation(self, calculation: str) -> str:
        """Process calculation with optional Chain-of-Thought generation."""
        if self.use_cot_generation:
            return self._generate_cot_breakdown(calculation)
        return calculation

    def _get_verification_steps(self, calculation: str) -> List[str]:
        """Break calculation into verification steps."""
        if self.use_step_breakdown:
            return self._extract_steps_from_cot(calculation)
        return [calculation]

    def _verify_steps(
        self, 
        instance: str, 
        instruction: str, 
        steps: List[str]
    ) -> List[str]:
        """Verify each step through variant generation and majority voting."""
        verified_steps = []
        context_builder = ContextBuilder(instance)
        
        for step in steps:
            # Build context from previous verified steps
            context = context_builder.build_context(verified_steps)
            
            # Generate variants for this step
            variants = self._generate_step_variants(
                instance, instruction, context, step
            )
            
            # Vote on best variant
            best_variant = self._select_best_variant(step, variants)
            verified_steps.append(best_variant)
            
        return verified_steps

    def _generate_cot_breakdown(self, calculation: str) -> str:
        """Generate Chain-of-Thought breakdown of calculation."""
        prompt = COT_PROMPT.format(calculation=calculation)
        return self.engine(prompt)

    def _extract_steps_from_cot(self, cot_text: str) -> List[str]:
        """Extract individual steps from Chain-of-Thought text."""
        # Try to extract steps marked with <STEP> tags
        step_pattern = r"<STEP>(.*?)</STEP>"
        steps = re.findall(step_pattern, cot_text, re.DOTALL)
        steps = [step.strip() for step in steps if step.strip()]
        
        # Fallback: split by lines if no tags found
        if not steps:
            steps = self._fallback_step_extraction(cot_text)
            
        return steps

    def _fallback_step_extraction(self, text: str) -> List[str]:
        """Fallback method to extract steps when no tags are present."""
        lines = text.split('\n')
        return [
            line.strip() for line in lines 
            if line.strip() and len(line.strip()) > 10
        ]

    def _generate_step_variants(
        self, 
        instance: str, 
        instruction: str, 
        context: str, 
        step: str
    ) -> List[str]:
        """Generate multiple variants for verification of a single step."""
        variants = []
        
        for i, verification_prompt in enumerate(self.verification_task_prompts):
            if self.enable_logging:
                print(f"Generating variant {i + 1}...")
                
            variant = self._generate_single_variant(
                instance, instruction, context, step, verification_prompt
            )
            variants.append(variant)
            
        return variants

    def _generate_single_variant(
        self, 
        instance: str, 
        instruction: str, 
        context: str, 
        step: str, 
        verification_prompt: str
    ) -> str:
        """Generate a single verification variant."""
        # Don't include context if not using step breakdown
        if not self.use_step_breakdown:
            context = ""
            
        prompt = VARIANT_GENERATION_PROMPT.format(
            instance=instance,
            instruction=instruction,
            previous_context=context,
            calculation=step,
            verification_task_prompt=verification_prompt
        )
        
        return self.engine(prompt)

    def _select_best_variant(self, original_step: str, variants: List[str]) -> str:
        """Use majority voting to select the best variant."""
        formatted_variants = self._format_variants_for_voting(variants)
        
        voting_prompt = MAJORITY_VOTING_PROMPT.format(
            calculation=original_step,
            generated_variants=formatted_variants
        )
        
        return self.engine(voting_prompt)

    def _format_variants_for_voting(self, variants: List[str]) -> str:
        """Format variants for the voting prompt."""
        return "\n".join(
            f"Variant {i + 1}: ```{variant}```" 
            for i, variant in enumerate(variants)
        )

    def _format_verified_result(self, verified_steps: List[str]) -> str:
        """Format the final verified result with proper tags."""
        return "\n".join(
            f"<VERIFIED>{step}</VERIFIED>" 
            for step in verified_steps
        )


class ContextBuilder:
    """Helper class to build context from previous verification steps."""
    
    def __init__(self, instance: str):
        self.base_context = f"These are previous context to help you verify calculation:\n{instance}\n"
    
    def build_context(self, verified_steps: List[str]) -> str:
        """Build context string from verified steps."""
        if not verified_steps:
            return self.base_context
            
        step_context = "".join(
            f"Step {i + 1}: ```{step}```" 
            for i, step in enumerate(verified_steps)
        )
        
        return self.base_context + step_context