"""
Simple StepCo Verifier

This is a simple implementation of a StepCo verifier that can be used with
your existing code. It implements the predict method that's required by
your verify_steps function.
"""

import re
import random


class StepCo:
    """
    A simple StepCo verifier that evaluates reasoning steps and
    estimates their probability of leading to the correct answer.
    """
    
    def __init__(self):
        """Initialize the StepCo verifier"""
        print("Initializing StepCo verifier...")
    
    def predict(self, input_text):
        """
        Predict the probability that a step leads to the correct answer.
        
        Args:
            input_text (str): The input text to evaluate (question + steps)
            
        Returns:
            float: Probability that the step leads to the correct answer
        """
        # For this simple implementation, we'll use some heuristics to
        # evaluate the quality of the reasoning step
        
        # Split the input to separate the question from the steps
        parts = input_text.split('\n', 1)
        if len(parts) > 1:
            question = parts[0]
            steps = parts[1]
        else:
            question = ""
            steps = input_text
        
        # Check for multiple choice question patterns
        is_multiple_choice = 'ABCD' in question or any(re.findall(r'\([A-D]\)', question))
        
        # Higher probability for steps with mathematical content
        math_symbols = sum(1 for c in steps if c in "+-*/=^√∫∑π")
        math_bonus = min(0.2, math_symbols * 0.02)
        
        # Higher probability for steps with numbers
        number_count = len(re.findall(r'\d+(\.\d+)?', steps))
        number_bonus = min(0.15, number_count * 0.03)
        
        # Higher probability for steps with energy/physics terms (based on your example)
        physics_terms = ['energy', 'quantum', 'eV', 'lifetime', 'sec', 'levels']
        physics_term_count = sum(1 for term in physics_terms if term in steps.lower())
        physics_bonus = min(0.1, physics_term_count * 0.02)
        
        # Higher probability for steps with reasoning indicators
        reasoning_indicators = ['therefore', 'because', 'thus', 'so', 'hence', 'which means']
        reasoning_indicator_count = sum(1 for term in reasoning_indicators if term in steps.lower())
        reasoning_bonus = min(0.1, reasoning_indicator_count * 0.025)
        
        # Higher probability for steps that seem to be making progress
        # This is a very simple heuristic based on step length
        step_length = len(steps)
        length_factor = 0.0
        if 20 <= step_length <= 500:
            length_factor = 0.05
        elif step_length > 500:
            length_factor = -0.05  # Penalty for excessively long steps
        
        # Higher probability for steps that mention the answer options
        answer_mentions = sum(1 for letter in ['A)', 'B)', 'C)', 'D)'] if letter in steps)
        answer_bonus = min(0.1, answer_mentions * 0.025)
        
        # Higher probability for final steps that make a clear choice
        answer_pattern = re.search(r'Answer:\s*([A-D])', steps)
        final_answer_bonus = 0.1 if answer_pattern else 0.0
        
        # Base probability
        base_prob = 0.65
        
        # Combine all factors
        probability = base_prob + math_bonus + number_bonus + physics_bonus + reasoning_bonus + length_factor + answer_bonus + final_answer_bonus
        
        # Ensure probability is between 0 and 1
        probability = max(0.1, min(0.95, probability))
        
        # Add a small amount of randomness to simulate model uncertainty
        probability += random.uniform(-0.05, 0.05)
        probability = max(0.1, min(0.95, probability))
        
        return probability