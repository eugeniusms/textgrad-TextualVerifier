from textgrad.engine import get_engine
from textgrad import Variable
from textgrad.optimizer import TextualGradientDescent
from textgrad.loss import TextLoss
from dotenv import load_dotenv
load_dotenv()

# Introduction: Variable
x = Variable("A sntence with a typo", role_description="The input sentence", requires_grad=True)
print(x.gradients)

# Introduction: Engine
# When we talk about the engine in TextGrad, we are referring to an LLM. 
# The engine is an abstraction we use to interact with the model.
engine = get_engine("gemini-pro")

# This object behaves like you would expect an LLM to behave: 
# You can sample generation from the engine using the generate method.
print(engine.generate("Hello how are you?"))