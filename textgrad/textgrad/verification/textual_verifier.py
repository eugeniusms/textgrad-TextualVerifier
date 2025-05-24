from typing import Union
import re
from textgrad.engine import EngineLM
from textgrad.config import validate_engine_or_get_default
from textgrad.verification.verifier import Verifier

class TextualVerifier(Verifier):
    """
    A verifier that uses an LLM to evaluate reasoning steps and 
    converts the textual evaluation to a probability score.
    """
    
    def __init__(self, engine: Union[str, EngineLM]):
        """
        Initialize the TextualVerifier verifier.
        
        Args:
            engine: The engine to use for evaluation
        """
        self.engine = validate_engine_or_get_default(engine)
    
    def verify(self, input_text: str) -> float:
        """
        Predict the probability that a step leads to the correct answer.
        
        Args:
            input_text (str): The input text to evaluate (question + steps)
            
        Returns:
            float: Probability that the step leads to the correct answer
        """
        print("Verifier: Textual")
        # Create a prompt for the LLM to evaluate the reasoning
        evaluation_prompt = self._create_evaluation_prompt(input_text)
        
        # Get the LLM's evaluation
        evaluation_result = self.engine(evaluation_prompt)
        
        # Extract a probability from the evaluation
        probability = self._extract_probability(evaluation_result)
        
        return probability
    
    def _create_evaluation_prompt(self, input_text: str) -> str:
        """
        Create a prompt for the LLM to evaluate the reasoning.
        
        Args:
            input_text (str): The input text to evaluate
            
        Returns:
            str: The evaluation prompt
        """
        # Split the input to separate the question from the steps
        parts = input_text.split('\n', 1)
        if len(parts) > 1:
            question = parts[0]
            reasoning_steps = parts[1]
        else:
            question = ""
            reasoning_steps = input_text
        
        # Create the prompt
        prompt = f"""
        As an expert evaluator, I need you to assess the quality of the following reasoning step for solving a problem.

        PROBLEM:
        {question}

        REASONING STEPS:
        {reasoning_steps}

        Evaluate the quality of this reasoning on a scale of 0 to 100, where:
        - 0-20: The reasoning contains critical errors or misconceptions
        - 21-40: The reasoning has significant flaws but some valid elements
        - 41-60: The reasoning is somewhat correct but incomplete or imprecise
        - 61-80: The reasoning is mostly correct with minor flaws
        - 81-100: The reasoning is excellent, clear, and correct

        First, analyze the reasoning steps critically, identifying any strengths or weaknesses.
        Then provide a final score between 0 and 100.

        Your response should follow this format:
        ANALYSIS: [Your detailed analysis of the reasoning]
        SCORE: [A number between 0 and 100]
        """
        return prompt
    
    def _extract_probability(self, evaluation_result: str) -> float:
        """
        Extract a probability from the LLM's evaluation.
        
        Args:
            evaluation_result (str): The LLM's evaluation
            
        Returns:
            float: A probability between 0 and 1
        """
        # Try to extract a score from the evaluation
        score_match = re.search(r'SCORE:\s*(\d+)', evaluation_result)
        
        if score_match:
            # Convert the score (0-100) to a probability (0-1)
            score = int(score_match.group(1))
            probability = score / 100.0
        else:
            # If no score is found, try to extract a number from the text
            number_match = re.search(r'(\d+)(?:\s*\/\s*100|\s*percent|\s*%)', evaluation_result)
            if number_match:
                score = int(number_match.group(1))
                probability = score / 100.0
            else:
                # Default to a moderate probability if no score is found
                probability = 0.5
        
        # Ensure the probability is between 0.1 and 0.95
        probability = max(0.1, min(0.95, probability))
        
        return probability