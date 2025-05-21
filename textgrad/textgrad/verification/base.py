from abc import ABC, abstractmethod
from textgrad.engine import EngineLM
from textgrad.variable import Variable

class BaseVerifier(ABC):
    def __init__(self, engine: EngineLM):
        self.engine = engine
        
    @abstractmethod
    def verify_update(self, original_variable: Variable, new_value: str, objective, context=None):
        """
        Verify if the proposed update to the variable is valid.
        
        Args:
            original_variable: The variable being updated
            new_value: The proposed new value for the variable
            objective: The optimization objective
            context: Additional context information
            
        Returns:
            Tuple of (is_valid, confidence, corrections)
        """
        pass

def get_verifier(strategy, engine):
    """Factory function to get the appropriate verifier"""
    if strategy == "process":
        from .process_verification import ProcessVerifier
        return ProcessVerifier(engine)
    elif strategy == "outcome":
        from .outcome_verification import OutcomeVerifier
        return OutcomeVerifier(engine)
    else:
        raise ValueError(f"Unknown verification strategy: {strategy}")