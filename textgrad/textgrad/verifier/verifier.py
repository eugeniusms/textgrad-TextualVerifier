from textgrad.variable import Variable

class Verifier:
    """Base class for verification models that evaluate calculations."""
    
    def verify(self, instance: Variable, calculation: Variable) -> Variable:
        """
        Verify the calculation for the given instance.
        
        Args:
            instance: The variable to evaluate (ex: problem/question)
            calculation: Result that needs verification
            
        Returns:
            Variable: Verified and improved calculation
        """
        raise NotImplementedError("Subclasses must implement verify()")