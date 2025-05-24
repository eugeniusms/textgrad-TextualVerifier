import google.generativeai as genai
import os

def generate_llm_output(your_question, model_version="gemini-1.5-pro"):
    """
    Asks Gemini a question and lets you choose which Gemini version to use.

    Args:
        your_question (str): What you want to ask Gemini.
        model_version (str): Which Gemini to use, e.g., "gemini-pro",
                             "gemini-1.5-flash", or newer ones.

    Returns:
        str: Gemini's answer, or an error message.
    """
    # Get your secret key (API key) from your computer's settings
    # Make sure you've set an environment variable named GOOGLE_API_KEY
    # (e.g., export GOOGLE_API_KEY="YOUR_KEY_HERE" in your terminal)
    # api_key = os.getenv("GOOGLE_API_KEY")
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        return "Oops! I can't find your Google API Key. Please set it as an environment variable named GOOGLE_API_KEY."

    genai.configure(api_key=api_key)

    try:
        # Pick the Gemini version you want
        model = genai.GenerativeModel(model_name=model_version)

        # Ask Gemini your question
        response = model.generate_content(your_question)

        # Give back Gemini's answer
        return response.text

    except Exception as e:
        return f"Something went wrong: {e}"