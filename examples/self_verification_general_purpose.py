import textgrad as tg
from textgrad.variable import Variable
from textgrad.verification import GeneralPurposeVerifier
from textgrad.loss import VerificationLoss

# Set up the engines for verification and optimization
tg.set_backward_engine("gemini-1.5-pro")
verification_engine = tg.get_engine("gemini-1.5-pro")
reasoning_engine = tg.get_engine("gemini-1.5-pro")

initial_solution = """To solve the equation 3x^2 - 7x + 2 = 0, we use the quadratic formula:
x = (-b ± √(b^2 - 4ac)) / 2a
a = 3, b = -7, c = 2
x = (7 ± √((-7)^2 + 4(3)(2))) / 6
x = (7 ± √73) / 6
The solutions are:
x1 = (7 + √73)
x2 = (7 - √73)"""

# Create the solution variable
solution = tg.Variable(initial_solution,
                       requires_grad=True,
                       role_description="solution to the math question")

# Create a system prompt for the loss function
loss_system_prompt = tg.Variable("""You will evaluate a solution to a math question. 
Do not attempt to solve it yourself, do not give a solution, only identify errors. Be super concise.""",
                              requires_grad=False,
                              role_description="system prompt")

# Create the loss function
loss = VerificationLoss(loss_system_prompt)
print(loss)

result = loss(solution)
print(result)