# Create textgrad/verification/hybrid_verification.py

from .base import BaseVerifier
from .process_verification import ProcessVerifier
from .outcome_verification import OutcomeVerifier
import re

class HybridSequentialVerifier(BaseVerifier):
    """A hybrid verification strategy that applies both process and outcome verification sequentially"""
    
    def __init__(self, engine):
        super().__init__(engine)
        self.process_verifier = ProcessVerifier(engine)
        self.outcome_verifier = OutcomeVerifier(engine)
        self.system_prompt = """You are a verification expert using a hybrid approach that considers both step-by-step reasoning processes and final outcomes. Your task is to provide a comprehensive verification by analyzing both how the variable was updated and whether the final result achieves the intended objective."""
    
    def verify_update(self, original_variable, new_value, objective, context=None):
        """Apply process verification first, then outcome verification if process verification passes"""
        
        # First apply process verification
        is_valid_process, confidence_process, corrections_process = self.process_verifier.verify_update(
            original_variable, new_value, objective, context
        )
        
        # If process verification fails with high confidence, return its result
        if not is_valid_process and confidence_process >= 0.7:
            return False, confidence_process, corrections_process
        
        # Then apply outcome verification
        is_valid_outcome, confidence_outcome, corrections_outcome = self.outcome_verifier.verify_update(
            original_variable, new_value, objective, context
        )
        
        # Merge results
        # If either verification passes with high confidence, consider it valid
        if (is_valid_process and confidence_process >= 0.7) or (is_valid_outcome and confidence_outcome >= 0.7):
            return True, max(confidence_process, confidence_outcome), None
        
        # If both fail, choose the correction with higher confidence
        if confidence_process > confidence_outcome:
            return False, confidence_process, corrections_process
        else:
            return False, confidence_outcome, corrections_outcome


class HybridCombinedVerifier(BaseVerifier):
    """A hybrid verification strategy that combines insights from both process and outcome verification"""
    
    def __init__(self, engine):
        super().__init__(engine)
        self.process_verifier = ProcessVerifier(engine)
        self.outcome_verifier = OutcomeVerifier(engine)
        self.system_prompt = """You are a verification expert using a combined approach that integrates insights from both step-by-step reasoning processes and final outcomes. Your task is to provide a comprehensive verification that balances concerns about the reasoning process with considerations about the final result."""
    
    def verify_update(self, original_variable, new_value, objective, context=None):
        """Apply both process and outcome verification and combine the results"""
        
        # Apply both verification strategies
        is_valid_process, confidence_process, corrections_process = self.process_verifier.verify_update(
            original_variable, new_value, objective, context
        )
        
        is_valid_outcome, confidence_outcome, corrections_outcome = self.outcome_verifier.verify_update(
            original_variable, new_value, objective, context
        )
        
        # Combine results using a weighted approach
        # Combine verification decisions
        if is_valid_process and is_valid_outcome:
            is_valid_combined = True
            confidence_combined = 0.5 * (confidence_process + confidence_outcome)
            corrections_combined = None
        elif not is_valid_process and not is_valid_outcome:
            is_valid_combined = False
            confidence_combined = 0.5 * (confidence_process + confidence_outcome)
            # Use corrections from the higher confidence verification
            corrections_combined = corrections_process if confidence_process > confidence_outcome else corrections_outcome
        else:
            # Disagreement between verifiers
            # If one has much higher confidence, go with that one
            if abs(confidence_process - confidence_outcome) > 0.2:
                if confidence_process > confidence_outcome:
                    is_valid_combined = is_valid_process
                    confidence_combined = confidence_process
                    corrections_combined = corrections_process
                else:
                    is_valid_combined = is_valid_outcome
                    confidence_combined = confidence_outcome
                    corrections_combined = corrections_outcome
            else:
                # If confidences are close, resolve the disagreement
                verification_prompt = self._build_disagreement_resolution_prompt(
                    original_variable.value,
                    new_value,
                    objective,
                    original_variable.get_role_description(),
                    context,
                    is_valid_process, confidence_process, corrections_process,
                    is_valid_outcome, confidence_outcome, corrections_outcome
                )
                
                # Get resolution from the verification engine
                resolution_result = self.engine(verification_prompt, system_prompt=self.system_prompt)
                
                # Parse the resolution result
                is_valid_combined, confidence_combined, corrections_combined = self._parse_resolution_result(resolution_result)
        
        return is_valid_combined, confidence_combined, corrections_combined
    
    def _build_disagreement_resolution_prompt(self, original_value, new_value, objective, role_description, context,
                                              is_valid_process, confidence_process, corrections_process,
                                              is_valid_outcome, confidence_outcome, corrections_outcome):
        context_text = f"Additional context:\n{context}\n\n" if context else ""
        
        process_result = f"VALID (confidence: {confidence_process})" if is_valid_process else f"INVALID (confidence: {confidence_process})"
        outcome_result = f"VALID (confidence: {confidence_outcome})" if is_valid_outcome else f"INVALID (confidence: {confidence_outcome})"
        
        process_corrections = "No corrections needed." if corrections_process is None else corrections_process
        outcome_corrections = "No corrections needed." if corrections_outcome is None else corrections_outcome
        
        return f"""
        There is a disagreement between process verification and outcome verification for the following update.
        {context_text}
        
        Role of the variable: {role_description}
        Original value: {original_value}
        Updated value: {new_value}
        
        Objective: {objective}
        
        Process verification result: {process_result}
        Process verification corrections: {process_corrections}
        
        Outcome verification result: {outcome_result}
        Outcome verification corrections: {outcome_corrections}
        
        Please carefully analyze both verification results and resolve the disagreement. Consider:
        1. Which verification approach is more relevant to the specific update being verified
        2. Which verification approach has identified more concrete issues (if any)
        3. Whether the issues identified are critical or minor
        
        Format your resolution as follows:
        
        <RESOLUTION_ANALYSIS>
        [Your analysis of the disagreement and justification for your resolution]
        </RESOLUTION_ANALYSIS>
        
        <RESOLUTION>
        Valid: [YES or NO]
        Confidence: [A number between 0.0 and 1.0]
        </RESOLUTION>
        
        <CORRECTIONS>
        [If you determine the update is invalid (Valid: NO), provide specific corrections here. If valid, write "No corrections needed."]
        </CORRECTIONS>
        """
    
    def _parse_resolution_result(self, resolution_result):
        # Extract verification decision
        valid_match = re.search(r'<RESOLUTION>\s*Valid:\s*(YES|NO)', resolution_result, re.IGNORECASE)
        is_valid = False
        if valid_match:
            is_valid = valid_match.group(1).upper() == "YES"
        
        # Extract confidence score
        confidence_match = re.search(r'Confidence:\s*([0-9]*\.?[0-9]+)', resolution_result)
        confidence = 0.5  # Default confidence
        if confidence_match:
            confidence = float(confidence_match.group(1))
        
        # Extract corrections if any
        corrections = None
        if not is_valid:
            corrections_match = re.search(r'<CORRECTIONS>\s*(.*?)\s*</CORRECTIONS>', resolution_result, re.DOTALL)
            if corrections_match:
                corrections_text = corrections_match.group(1).strip()
                if corrections_text != "No corrections needed.":
                    corrections = corrections_text
        
        return is_valid, confidence, corrections