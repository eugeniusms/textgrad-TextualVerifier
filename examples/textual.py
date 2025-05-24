import textgrad as tg
from textgrad.engine import get_engine
from textgrad.variable import Variable
from textgrad.optimizer import TextualGradientDescent
from textgrad.verifier import TextualVerifier
from textgrad.loss import TextLoss

engine = get_engine("gemini-1.5-pro")
tg.set_backward_engine("gemini-1.5-pro")

initial_solution = """To solve the equation 3x^2 - 7x + 2 = 0, we use the quadratic formula:
x = (-b ± √(b^2 - 4ac)) / 2a
a = 3, b = -7, c = 2
x = (7 ± √((-7)^2 + 4(3)(2))) / 6
x = (7 ± √73) / 6
The solutions are:
x1 = (7 + √73)
x2 = (7 - √73)"""

solution = Variable(initial_solution,
                       requires_grad=True,
                       role_description="solution to the math question")

loss_system_prompt = Variable("""You will evaluate a solution to a math question. 
Do not attempt to solve it yourself, do not give a solution, only identify errors. Be super concise.""",
                              requires_grad=False,
                              role_description="system prompt")

optimizer = TextualGradientDescent([solution])
optimizer.zero_grad()

loss = TextLoss(loss_system_prompt, engine=engine)
result = loss(solution) # Forward method in Loss Function

print("INITIAL LOSS:", result)

# Verify Loss
verifier = TextualVerifier(verifier_engine=engine, step_eval_iterations=3)
verified_result = verifier.verify(instance=solution, 
                                    prompt=loss_system_prompt,
                                    calculation=result)

print("FINAL LOSS:", verified_result)

# Optimize
verified_result.backward()
optimizer.step()
print(solution.value)