## DEFAULT
DEFAULT_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
DEFAULT_PROMPTS = [
    {"prompt": "What is the sum of 2 and 3?"},
    {"prompt": "Explain Newton's second law of motion."}
] 

## TACO
def generate_prompt(test_case, prompt, starter_code=None):
    """Generate a prompt for the LLM to solve a problem."""
    formatted_prompt = ""

    data = test_case
    if not data.get("fn_name"):
        formatted_prompt += "Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition."  # noqa
    else:
        formatted_prompt += (
            "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution."  # noqa
        )

    data = prompt
    formatted_prompt += data
    if starter_code is not None:
        data = starter_code
        data = "\n" + data  # + "\n"
        formatted_prompt += data

    return formatted_prompt

## AIME
AIME_USER_PROMPT = """{problem}"""

## GPQA
GPQA_USER_PROMPT = """{question}"""