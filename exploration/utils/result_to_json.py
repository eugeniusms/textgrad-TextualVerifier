import json

def format_reasoning_to_json(reasoning_data, filename="reasoning_output.json"):
    initial_steps = reasoning_data["initial_steps"]
    final_steps = reasoning_data["final_steps"]

    initial_steps_formatted = {f"Step {i+1}": step for i, step in enumerate(initial_steps)}
    final_steps_formatted = {f"Step {i+1}": step for i, step in enumerate(final_steps)}

    # Create the final output dictionary
    output = {
        "initial_steps": initial_steps_formatted,
        "final_steps": final_steps_formatted,
        "final_answer": reasoning_data["answer"]
    }

    # Write to a JSON file with indentation
    with open(filename, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Output successfully written to {filename}")