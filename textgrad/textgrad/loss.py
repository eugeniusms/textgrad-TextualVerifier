import re
from textgrad.engine import EngineLM, get_engine
from textgrad.variable import Variable
from typing import List, Union
from textgrad.autograd import LLMCall, FormattedLLMCall, OrderedFieldsMultimodalLLMCall
from textgrad.autograd import Module
from .config import SingletonBackwardEngine
from textgrad.verification import Verifier, TextualVerifier # NEW VERIFICATION


class TextLoss(Module):
    def __init__(self, 
                 eval_system_prompt: Union[Variable, str],
                 engine: Union[EngineLM, str] = None):
        """
        A vanilla loss function to evaluate a response.
        In particular, this module is used to evaluate any given text object.

        :param evaluation_instruction: The evaluation instruction variable.
        :type evaluation_instruction: Variable
        :param engine: The EngineLM object.
        :type engine: EngineLM
        
        :example:
        >>> from textgrad import get_engine, Variable
        >>> from textgrad.loss import TextLoss
        >>> engine = get_engine("gpt-4o")
        >>> evaluation_instruction = Variable("Is ths a good joke?", requires_grad=False)
        >>> response_evaluator = TextLoss(evaluation_instruction, engine)
        >>> response = Variable("What did the fish say when it hit the wall? Dam.", requires_grad=True)
        >>> response_evaluator(response)
        """
        super().__init__()
        if isinstance(eval_system_prompt, str):
            eval_system_prompt = Variable(eval_system_prompt, requires_grad=False, role_description="system prompt for the evaluation")
        self.eval_system_prompt = eval_system_prompt
        if ((engine is None) and (SingletonBackwardEngine().get_engine() is None)):
            raise Exception("No engine provided. Either provide an engine as the argument to this call, or use `textgrad.set_backward_engine(engine)` to set the backward engine.")
        elif engine is None:
            engine = SingletonBackwardEngine().get_engine()
        if isinstance(engine, str):
            engine = get_engine(engine)
        self.engine = engine
        self.llm_call = LLMCall(self.engine, self.eval_system_prompt)

    def forward(self, instance: Variable):
        """
        Calls the ResponseEvaluation object.

        :param instance: The instance variable.
        :type instance: Variable
        :return: The result of the evaluation
        """
        return self.llm_call(instance)

class MultiFieldEvaluation(Module):
    def __init__(
        self,
        evaluation_instruction: Variable,
        role_descriptions: List[str],
        engine: Union[EngineLM, str] = None,
        system_prompt: Variable = None,
    ):
        """A module to compare two variables using a language model.

        :param evaluation_instruction: Instruction to use as prefix for the comparison, specifying the nature of the comparison.
        :type evaluation_instruction: Variable
        :param engine: The language model to use for the comparison.
        :type engine: EngineLM
        :param v1_role_description: Role description for the first variable, defaults to "prediction to evaluate"
        :type v1_role_description: str, optional
        :param v2_role_description: Role description for the second variable, defaults to "correct result"
        :type v2_role_description: str, optional
        :param system_prompt: System prompt to use for the comparison, defaults to "You are an evaluation system that compares two variables."
        :type system_prompt: Variable, optional
        
        :example:
        TODO: Add an example
        """
        super().__init__()
        self.evaluation_instruction = evaluation_instruction
        if ((engine is None) and (SingletonBackwardEngine().get_engine() is None)):
            raise Exception("No engine provided. Either provide an engine as the argument to this call, or use `textgrad.set_backward_engine(engine)` to set the backward engine.")
        elif engine is None:
            engine = SingletonBackwardEngine().get_engine()
        if isinstance(engine, str):
            engine = get_engine(engine)
        self.engine = engine
        self.role_descriptions = role_descriptions
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = Variable("You are an evaluation system that compares two variables.",
                                            requires_grad=False,
                                            role_description="system prompt for the evaluation")
        format_string_items = ["{{instruction}}"]
        for role_description in role_descriptions:
            format_string_items.append(f"**{role_description}**: {{{role_description}}}")
        format_string = "\n".join(format_string_items)
        self.format_string = format_string.format(instruction=self.evaluation_instruction, **{role_description: "{"+role_description+"}" for role_description in role_descriptions})
        self.fields = {"instruction": self.evaluation_instruction, **{role_description: None for role_description in role_descriptions}}
        self.formatted_llm_call = FormattedLLMCall(engine=self.engine,
                                                   format_string=self.format_string,
                                                   fields=self.fields,
                                                   system_prompt=self.system_prompt)

    def forward(self, inputs: List[Variable]):
        for role_description, var in zip(self.role_descriptions, inputs):
            var.set_role_description(role_description)
        inputs_call = {"instruction": self.evaluation_instruction, 
                       **{role_description: var for role_description, var in zip(self.role_descriptions, inputs)}}
        return self.formatted_llm_call(inputs=inputs_call,
                                       response_role_description=f"evaluation of the a prediction")


class MultiFieldTokenParsedEvaluation(MultiFieldEvaluation):
    def __init__(
        self,
        evaluation_instruction: Variable,
        role_descriptions: List[str],
        engine: Union[EngineLM, str] = None,
        system_prompt: Variable = None,
        parse_tags: List[str] = None,
    ):
        super().__init__(
            evaluation_instruction=evaluation_instruction,
            role_descriptions=role_descriptions,
            engine=engine,
            system_prompt=system_prompt,
        )
        self.parse_tags = parse_tags

    def parse_output(self, response: Variable) -> str:
        """
        Parses the output response and returns the parsed response.

        :param response: The response to be parsed.
        :type response: Variable
        :return: The parsed response.
        :rtype: str
        """
        response_text = response.value
        response = response_text.split(self.parse_tags[0])[1].split(self.parse_tags[1])[0].strip()
        return response


DEFAULT_TEST_TIME = "You are an intelligent assistant used as an evaluator, and part of an optimization system. You will analyze a solution to a multi-choice problem. Investigate the reasoning and answer. Do not try to solve the problem, only raise the potential issues and mistakes in the answer. Be creative, think about different perspectives, and be very critical."

class MultiChoiceTestTime(Module):
    def __init__(self,
                 evaluation_instruction: str,
                 engine: Union[EngineLM, str] = None,
                 system_prompt: Variable = None):
        """
        The test-time loss to use when working on a response to a multiple choice question.

        :param evaluation_instruction: Instruction to guide the test time evaluation. This will be a prefix to the prompt.
        :type evaluation_instruction: str
        :param engine: LLM engine to use for the test-time loss computation.
        :type engine: EngineLM
        :param system_prompt: System prompt for the test-time loss computation, defaults to None
        :type system_prompt: Variable, optional
        """
        super().__init__()
        if system_prompt:
            self.tt_system_prompt = system_prompt
        else:
            tt_system_prompt = DEFAULT_TEST_TIME
            self.tt_system_prompt = Variable(tt_system_prompt,
                                                requires_grad=False,
                                                role_description="system prompt for the test-time evaluation")
        
        if ((engine is None) and (SingletonBackwardEngine().get_engine() is None)):
            raise Exception("No engine provided. Either provide an engine as the argument to this call, or use `textgrad.set_backward_engine(engine)` to set the backward engine.")
        elif engine is None:
            engine = SingletonBackwardEngine().get_engine()
        if isinstance(engine, str):
            engine = get_engine(engine)
        self.engine = engine
        format_string = "{instruction}\nQuestion: {{question}}\nAnswer by the language model: {{prediction}}"
        self.format_string = format_string.format(instruction=evaluation_instruction)
        self.fields = {"prediction": None, "question": None}
        self.formatted_llm_call = FormattedLLMCall(engine=self.engine,
                                                   format_string=self.format_string,
                                                   fields=self.fields,
                                                   system_prompt=self.tt_system_prompt)

    def forward(self, question: str, prediction: Variable) -> Variable:
        question_variable = Variable(question, 
                                     requires_grad=False, 
                                     role_description="the multiple choice question")

        inputs = {"prediction": prediction, "question": question_variable}
        return self.formatted_llm_call(inputs=inputs,
                                       response_role_description=f"evaluation of the {prediction.get_role_description()}")

class ImageQALoss(Module):
    def __init__(self,
                 evaluation_instruction: str,
                 engine: Union[EngineLM, str] = None,
                 system_prompt: Variable = None):
        super().__init__()
        self.evaluation_instruction = Variable(evaluation_instruction, role_description="evaluation instruction", requires_grad=False)
        if ((engine is None) and (SingletonBackwardEngine().get_engine() is None)):
            raise Exception("No engine provided. Either provide an engine as the argument to this call, or use `textgrad.set_backward_engine(engine)` to set the backward engine.")
        elif engine is None:
            engine = SingletonBackwardEngine().get_engine()
        if isinstance(engine, str):
            engine = get_engine(engine)
        self.engine = engine
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = Variable("You are an evaluation system that evaluates image-related questions.",
                                            requires_grad=False,
                                            role_description="system prompt for the evaluation")

        self.multimodal_llm_call = OrderedFieldsMultimodalLLMCall(engine=self.engine,
                                                                  system_prompt=self.system_prompt,
                                                                  fields=["Evaluation Instruction", "Question", "Image", "Answer"])

    def forward(self, image: Variable, question: Variable, response: Variable) -> Variable:
        
        inputs = {
            "Evaluation Instruction": self.evaluation_instruction,
            "Question": question,
            "Image": image,
            "Answer": response
        }
        return self.multimodal_llm_call(inputs=inputs,
                                        response_role_description=f"evaluation of the {response.get_role_description()}")


# NEW VERIFICATION
class VerifiedLoss(Module):
    def __init__(self, 
                 eval_system_prompt: Union[Variable, str],
                 engine: Union[EngineLM, str] = None,
                 verifier_engine: Union[EngineLM, str] = None,
                 verifier: Verifier = TextualVerifier,
                 threshold: float = 0.5,
                 max_revisions: int = 3):
        """
        A loss function that verifies and revises reasoning steps before evaluation.
        
        :param eval_system_prompt: System prompt for final evaluation
        :type eval_system_prompt: Union[Variable, str]
        :param engine: Engine for final evaluation and revision
        :type engine: Union[EngineLM, str]
        :param verifier_engine: Engine for verification (defaults to main engine)
        :type verifier_engine: Union[EngineLM, str]
        :param threshold: Probability threshold for accepting steps
        :type threshold: float
        :param max_revisions: Maximum revisions per step
        :type max_revisions: int
        
        :example:
        >>> from textgrad import get_engine, Variable
        >>> from textgrad.loss import VerificationLoss
        >>> engine = get_engine("gpt-4o")
        >>> eval_prompt = "Evaluate the quality of this reasoning"
        >>> verification_loss = VerificationLoss(eval_prompt, engine, threshold=0.7)
        >>> question = Variable("What is 2+2?", requires_grad=False)
        >>> reasoning = Variable("<Step>2+2=4</Step>", requires_grad=True)
        >>> result = verification_loss(question, reasoning)
        """
        super().__init__()
        
        # Setup evaluation prompt (following TextLoss pattern)
        if isinstance(eval_system_prompt, str):
            eval_system_prompt = Variable(eval_system_prompt, requires_grad=False, 
                                        role_description="system prompt for evaluation")
        self.eval_system_prompt = eval_system_prompt
        
        # Setup engines (following existing pattern)
        if ((engine is None) and (SingletonBackwardEngine().get_engine() is None)):
            raise Exception("No engine provided. Either provide an engine as the argument to this call, or use `textgrad.set_backward_engine(engine)` to set the backward engine.")
        elif engine is None:
            engine = SingletonBackwardEngine().get_engine()
        if isinstance(engine, str):
            engine = get_engine(engine)
        self.engine = engine
        
        # Setup verifier
        verifier_engine = verifier_engine or engine
        self.verifier = verifier(verifier_engine)
        
        # Verification parameters
        self.threshold = threshold
        self.max_revisions = max_revisions

    def forward(self, question: Variable) -> Variable:
        """
        Verify and revise reasoning steps, then evaluate the final result.
        
        This is where your verify_and_revise logic is implemented.
        
        :param question: The question being answered
        :type question: Variable
        :param reasoning_steps: The initial reasoning steps
        :type reasoning_steps: Variable
        :return: The evaluation of the verified reasoning
        :rtype: Variable
        """
        # Step 1: Extract steps from reasoning (using your step_formatter logic)
        reasoning_steps = self.cot_prompter(question)
        initial_steps = self._step_formatter(reasoning_steps)
        
        if not initial_steps:
            # If no steps found, treat the whole text as one step
            initial_steps = [reasoning_steps.value]
        
        print(f"Initial reasoning path with {len(initial_steps)} steps")
        for i, step in enumerate(initial_steps):
            print(f"Step {i+1}: {step[:100]}...")
        
        # Step 2: Verify and revise each step (using your verify_and_revise logic)
        final_steps = self._verify_and_revise(question.value, initial_steps)
        
        # Step 3: Create verified reasoning text
        verified_reasoning_text = "\n".join([f"<Step>{step}</Step>" for step in final_steps])
        
        # Step 4: Final evaluation using the eval_system_prompt
        evaluation_input = f"Question: {question.value}\n\nVerified Reasoning:\n{verified_reasoning_text}"
        
        # Create a formatted call for evaluation
        final_evaluation = self.engine(f"{self.eval_system_prompt.value}\n\n{evaluation_input}")
        
        return Variable(final_evaluation, requires_grad=True, 
                       role_description="evaluation of verified reasoning")

    def cot_prompter(self, query):
        initial_reasoning_path = f"""
            Mark the beginning and end of each reasoning step with <Step> 
            and </Step> tags. Q: q. A: Let's think step by step.

            {query}
        """
        return initial_reasoning_path
    
    def _step_formatter(self, reasoning_path: str) -> List[str]:
        """
        Extract individual steps from a reasoning path.
        This implements your step_formatter function.
        """
        # Use regex to extract content between <Step> and </Step> tags
        step_pattern = r"<Step>(.*?)</Step>"
        steps = re.findall(step_pattern, reasoning_path, re.DOTALL)
        
        # Clean up extracted steps
        return [step.strip() for step in steps]

    def _verify_and_revise(self, question: str, reasoning_chain: List[str]) -> List[str]:
        """
        Verify and revise each step in the reasoning chain.
        This implements your verify_and_revise function.
        """
        verified_chain = []
        
        # Process each step in the reasoning chain
        for i, step in enumerate(reasoning_chain):
            print(f"\nVerifying step {i+1}:")
            
            current_step = step
            
            # Try to revise this step until it meets the threshold or max revisions reached
            for revision in range(self.max_revisions):
                current_chain = verified_chain + [current_step]
                verification = self._verify_step(question, current_chain)
                
                print(f"Step {i+1} (Revision {revision}) probability: {verification['probability']:.4f}")
                
                # If step meets threshold, accept it and move to next step
                if verification['probability'] >= self.threshold:
                    print(f"Step {i+1} meets threshold, moving to next step")
                    verified_chain.append(current_step)
                    break
                    
                # If step doesn't meet threshold and we haven't reached max revisions, revise it
                if revision < self.max_revisions - 1:
                    print(f"Step {i+1} below threshold, revising...")
                    revised_step = self._revise_step(question, verified_chain, current_step, verification['probability'])
                    current_step = revised_step
                else:
                    # If we've reached max revisions, accept the current step and move on
                    print(f"Warning: Step {i+1} still below threshold after {self.max_revisions} revisions. Accepting anyway.")
                    verified_chain.append(current_step)
        
        return verified_chain

    def _verify_step(self, question: str, chain_so_far: List[str]) -> dict:
        """
        Verify a single step in the reasoning chain.
        This implements your verify_step function.
        """
        preceding_steps = "\n".join(chain_so_far)
        verifier_input = f"{question}\n{preceding_steps}"
        
        try:
            # Get probability that this step leads to correct answer
            probability = self.verifier.predict(verifier_input)
        except Exception as e:
            print(f"Error in verifier.predict(): {e}")
            probability = 0.5
        
        return {
            "probability": probability,
            "revise": probability < self.threshold
        }

    def _revise_step(self, question: str, previous_steps: List[str], 
                     current_step: str, probability: float) -> str:
        """
        Revise a single step in the reasoning chain.
        This implements your revise_step function.
        """
        # Format previous steps with tags
        previous_formatted = ""
        if previous_steps:
            previous_formatted = "\n".join([f"<Step>{step}</Step>" for step in previous_steps])
            previous_formatted = f"Previous steps:\n{previous_formatted}\n\n"
        
        # Create prompt for revising the step
        prompt = f"""Q: {question}
        
        {previous_formatted}Current step to revise:
        <Step>{current_step}</Step>

        The probability that this step leads to the correct answer is {probability:.4f}, which is below the acceptable threshold.
        Please revise ONLY this step to increase the probability that it leads to the correct answer.
        Your revision should be more accurate, clear, and logical. Make sure it follows directly from the previous steps.

        Provide your revised step between <Step> and </Step> tags."""
        
        # Generate & extract the revised step using the engine
        revised_output = self.engine(prompt)
        revised_steps = self._step_formatter(revised_output)
        
        # Return the revised step or the original if extraction failed
        if revised_steps and len(revised_steps) > 0:
            return revised_steps[0]
        else:
            return current_step
