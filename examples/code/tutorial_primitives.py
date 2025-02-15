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
engine = get_engine("gemini-1.5-pro")
print(engine.generate("Hello how are you?"))

# Introduction: Loss
system_prompt = Variable("Evaluate the correctness of this sentence", role_description="The system prompt")
loss = TextLoss(system_prompt, engine=engine)
print(loss)

# Introduction: Optimizer
optimizer = TextualGradientDescent(parameters=[x], engine=engine)

# Putting it all together
l = loss(x)
l.backward(engine)
optimizer.step()

print(x.value)

# Multiple Optimization
optimizer.zero_grad()