from textgrad.variable import Variable

class Verifier:
    """Base class for verification models that evaluate the calculation for instance."""
    
    def verify(self, instance: Variable, calculation: Variable) -> str:
        """
        Verify the calculation of instance.
        
        Args:
            instance: The variable to evaluate (ex: solution/prompt)
            calculation: Result calculated by loss/optimizer function
            
        Returns:
            str: Verification result (updated calculation)
        """
        raise NotImplementedError("Subclasses must implement verify()")

    