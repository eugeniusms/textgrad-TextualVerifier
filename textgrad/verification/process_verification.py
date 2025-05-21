from .base import BaseVerifier
from textgrad.variable import Variable
import re

class ProcessVerifier(BaseVerifier):
    def __init__(self, engine):
        super().__init__(engine)
        self.system_prompt = """You are a verification expert specializing in analyzing step-by-step reasoning. Your task is to verify if updates to variables are valid and improve the objective. You make thorough, logical evaluations of each step in the reasoning process."""
    
    def verify_update(self, original_variable, new_value, objective, context=None):
        """Implement process-supervised verification based on 'Let's Verify Step by Step'"""
        verification_prompt = self._build_verification_prompt(
            original_variable.value, 
            new_value,
            objective,
            original_variable.get_role_description(),
            context
        )
        
        # Get verification result from the verification engine
        verification_result = self.engine(verification_prompt, system_prompt=self.system_prompt)
        
        # Parse the verification result
        is_valid, confidence, corrections = self._parse_verification_result(verification_result)
        
        return is_valid, confidence, corrections
    
    def _build_verification_prompt(self, original_value, new_value, objective, role_description, context):
        context_text = f"Additional context:\n{context}\n\n" if context else ""
        
        return f"""
        Your task is to verify if the updated text is valid and improves upon the original according to the objective.
        {context_text}
        Role of the variable: {role_description}

        Original value: {original_value}
        Updated value: {new_value}

        Objective: {objective}

        Step 1: Break down the update into individual reasoning steps or changes.
        Step 2: For each step or change, verify if it correctly addresses the objective.
        Step 3: Identify any hallucinations, errors in reasoning, or invalid claims.
        Step 4: Provide a final verification decision with confidence score and corrections if needed.

        Format your response as follows:

        <ANALYSIS>
        [Your step-by-step analysis of the changes and verification]
        </ANALYSIS>

        <VERIFICATION>
        Valid: [YES or NO]
        Confidence: [A number between 0.0 and 1.0]
        </VERIFICATION>

        <CORRECTIONS>
        [If the update is invalid (Valid: NO), provide corrections here. If valid, write "No corrections needed."]
        </CORRECTIONS>
        """
    
    def _parse_verification_result(self, verification_result):
        # Extract verification decision
        valid_match = re.search(r'<VERIFICATION>\s*Valid:\s*(YES|NO)', verification_result, re.IGNORECASE)
        is_valid = False
        if valid_match:
            is_valid = valid_match.group(1).upper() == "YES"
        
        # Extract confidence score
        confidence_match = re.search(r'Confidence:\s*([0-9]*\.?[0-9]+)', verification_result)
        confidence = 0.5  # Default confidence
        if confidence_match:
            confidence = float(confidence_match.group(1))
        
        # Extract corrections if any
        corrections = None
        if not is_valid:
            corrections_match = re.search(r'<CORRECTIONS>\s*(.*?)\s*</CORRECTIONS>', verification_result, re.DOTALL)
            if corrections_match:
                corrections_text = corrections_match.group(1).strip()
                if corrections_text != "No corrections needed.":
                    corrections = corrections_text
        
        return is_valid, confidence, corrections