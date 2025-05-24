import textgrad as tg
from textgrad.variable import Variable
from textgrad.verification import GeneralPurposeVerifier
from textgrad.loss import TextLoss, VerifiedLoss

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

optimizer = tg.TGD([solution])

# TextLoss Basic
loss1 = TextLoss(loss_system_prompt)
result1 = loss1(solution)
print(result1)

result1.backward()
optimizer.step()
print(solution.value)

# Verification Loss
optimizer.zero_grad()

loss2 = VerifiedLoss(eval_system_prompt=loss_system_prompt, 
                        verifier=GeneralPurposeVerifier,
                        threshold=0.7,
                        max_revisions=3)
result2 = loss2(solution)
print(result2)

result2.backward()
optimizer.step()
print(solution.value)