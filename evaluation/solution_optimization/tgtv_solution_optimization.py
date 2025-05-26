import re
import json
import argparse
import concurrent
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv
load_dotenv(override=True)
from statistics import multimode

import textgrad as tg
from textgrad.tasks import load_instance_task
from textgrad.verifier import TextualVerifier


def config():
    parser = argparse.ArgumentParser(description="Optimize a prompt for a task.")
    parser.add_argument("--task", type=str, default="MMLU_machine_learning", help="The task to evaluate the model on.")
    parser.add_argument("--engine", type=str, default="gemini-1.5-pro", help="The API to use for evaluation.")
    parser.add_argument("--max_iterations", type=int, default=3, help="The maximum number of iterations of test-time updates.")
    parser.add_argument("--num_threads", type=int, default=16, help="The number of threads to use for evaluation.")
    parser.add_argument("--use_verifier", action="store_true", help="Enable TextualVerifier for loss verification.")
    parser.add_argument("--verifier_iterations", type=int, default=3, help="Number of verification iterations.")
    return parser.parse_args()


class MajorityVoting:
    def __init__(self):
        pass

    def __call__(self, predictions):
        ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*([A-D])"
        pred_labels = []
        for pred in predictions:
            match = re.search(ANSWER_PATTERN_MULTICHOICE, pred.value)
            extracted_answer = match.group(1) if match else None
            pred_labels.append(extracted_answer)
        
        modes = multimode(pred_labels)
        return tg.Variable(f"Answer: {modes[0]}", role_description="Majority ensemble")


def get_zeroshot_answer(question):
    """Getting the zero-shot answer from an LLM without optimizing the response at test time."""
    # The system prompt is from: https://github.com/openai/simple-evals/blob/main/sampler/chat_completion_sampler.py
    STARTING_SYSTEM_PROMPT = (
    "You are Gemini, a large language model trained by Google, based on the Gemini architecture."
    + "\nKnowledge cutoff: 2023-12\nCurrent date: 2024-04-01"
)
    system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT, requires_grad=False, role_description="system prompt to the language model")
    model = tg.BlackboxLLM(llm_engine, system_prompt)
    response = model(tg.Variable(question, requires_grad=False, role_description="question to the language model"))
    return response

def run_test_time_training(sample):
    performance_history = []
    question, answer, test_time_objective, instance_eval_fn = sample
    zero_shot_response = get_zeroshot_answer(question)
    
    instance_var = tg.Variable(zero_shot_response.value,
                               requires_grad=True,
                               role_description="creative and precise solution and the prediction for the multiple choice question")
    
    # Evaluate the zero-shot response
    performance_history.append(int(instance_eval_fn(instance_var)))
    
    optimizer = tg.TextualGradientDescent(engine=llm_engine, 
                                          parameters=[instance_var], 
                                          constraints=["The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD."])

    predictions = []
    predictions.append(tg.Variable(
        instance_var.value,
        role_description=instance_var.role_description
        ))

    # Initialize verifier if enabled
    verifier = None
    if args.use_verifier:
        verifier = TextualVerifier(verifier_engine=llm_engine, 
                                 step_eval_iterations=args.verifier_iterations)

    # Start test time training
    for iteration in range(args.max_iterations):
        optimizer.zero_grad()
        
        test_time_loss = test_time_objective(instance_var)
        
        if args.use_verifier and verifier:
            question_var = tg.Variable(question, requires_grad=False, role_description="original question")
            
            # Create loss prompt variable (extract from the test_time_objective if possible)
            loss_prompt = tg.Variable(
                "Evaluate this answer to the multiple choice question. Identify any errors or areas for improvement.",
                requires_grad=False,
                role_description="evaluation prompt"
            )
            
            try:
                # Verify the loss calculation
                verified_result = verifier.verify(instance=question_var,
                                                prompt=loss_prompt,
                                                calculation=test_time_loss)
                
                # Transfer the verified result back to the loss
                test_time_loss.set_value(verified_result.value)
                
            except Exception as e:
                print(f"⚠️ Verification failed: {e}, continuing with original loss")
        
        # Backward pass and optimization step
        test_time_loss.backward()
        optimizer.step()
        
        current_performance = instance_eval_fn(instance_var)
        performance_history.append(current_performance)
        
        predictions.append(tg.Variable(
            instance_var.value,
            role_description=instance_var.role_description
        ))

    # Create ensemble prediction
    ensembled_prediction = ensembler(predictions)
    ensemble_performance = instance_eval_fn(ensembled_prediction)
    performance_history.append(ensemble_performance)
    predictions.append(ensembled_prediction)

    return performance_history, predictions, question, answer


args = config()
llm_engine = tg.get_engine(engine_name=args.engine)
tg.set_backward_engine(llm_engine, override=True)
test_set = load_instance_task(args.task, evaluation_api=llm_engine)
ensembler = MajorityVoting()

all_solutions = {}
with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
    futures = []
    for _, sample in enumerate(test_set):
        future = executor.submit(run_test_time_training, sample)
        futures.append(future)

    all_history = []
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=0):
        performance_history, predictions, question, answer = future.result()
        all_solutions[question] = {"predictions": [p.value for p in predictions], "answer": answer}
        all_history.append(performance_history)

print(np.array(all_history).mean(axis=0))
with open(f"./{args.task}_predictions.json", "w") as f:
    json.dump(all_solutions, f)