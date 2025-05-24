from textgrad.variable import Variable
from typing import Union
from textgrad.engine import EngineLM

def verify(instance: Variable, 
          calculation: Variable, 
          verifier, 
          verifier_engine: Union[str, EngineLM] = None,
          **verifier_kwargs) -> Variable:
    """
    Basic verify method that takes an instance and calculation,
    then uses a verifier to improve the calculation.
    
    Args:
        instance: The original problem/question (Variable)
        calculation: The result that needs verification (Variable) 
        verifier: The verifier class to use (e.g., TextualVerifier)
        verifier_engine: Engine for the verifier
        **verifier_kwargs: Additional arguments for verifier
        
    Returns:
        Variable: Verified and improved calculation
    """
    
    # Step 1: Create the verifier instance
    if verifier_engine is not None:
        verifier_instance = verifier(engine=verifier_engine, **verifier_kwargs)
    else:
        verifier_instance = verifier(**verifier_kwargs)
    
    # Step 2: Run verification 
    verified_result = verifier_instance.verify(instance, calculation)
    
    # Step 3: Return the verified result
    return verified_result