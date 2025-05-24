class Verifier:
    """Base class for verification models that evaluate reasoning steps."""
    
    def predict(self, input_text: str) -> float:
        """
        Predict the probability that a step leads to the correct answer.
        
        Args:
            input_text (str): The input text to evaluate (question + steps)
            
        Returns:
            float: Probability that the step leads to the correct answer
        """
        raise NotImplementedError("Subclasses must implement predict()")

    