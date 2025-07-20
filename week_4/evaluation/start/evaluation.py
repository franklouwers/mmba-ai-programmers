import json
from pprint import pprint
from openai import OpenAI
from langsmith.evaluation import evaluate

# Initialize the OpenAI client
client = OpenAI()

# Dataset name in LangSmith (already uploaded)
dataset_name = "news_extraction_homework"

# System prompt for extraction
SYSTEM_PROMPT = """Extract information from the news into a dictionary. 
The dictionary keys are company_name, date_of_transaction, amount, product_service, location (formatted as 'city, state' where state is 2 letters in capitals). 
Make sure the output is in proper JSON with double quotes around the keys and values. 
For all dates, use the following format: mm-dd-yyyy.

It is CRUCIAL and non-negotiable that the ouput is pure json. Not ```json-like structures. Pure json. No extra fluff around it.
"""


def make_call_to_llm(input):
    # Extract the input content from the dataset item
    user_content = input["news"] if isinstance(input, dict) else input

    # Create the message array for the API call
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

    # Call the OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=messages,
        temperature=0
    )

    # Extract the output from the response
    output = response.choices[0].message.content

    return {"output": output}

# llm_result is the result of the make_call_to_llm call
# It contains the model's response that we'll compare against the expected output


def perform_eval(llm_result, dataset_item):
    try:
        # Parse the model's output
        llm_output = json.loads(llm_result.outputs['output'])

        # Parse the expected output
        expected_output = json.loads(dataset_item.outputs['output'])

        # Extract score from response
        # For a simpler implementation, let's manually calculate the score
        total_keys = len(expected_output)
        correct_keys = sum(1 for key in expected_output if key in llm_output and llm_output[key] == expected_output[key])
        score = correct_keys / total_keys if total_keys > 0 else 0

        return {"score": score}
    except json.JSONDecodeError:
        # Handle the case where JSON parsing fails
        return {"score": 0.0}


evaluate(
    make_call_to_llm,   # function to call the LLM
    data=dataset_name,  # dataset name already stored in LangChain
    evaluators=[
        perform_eval    # function used to evaluate
    ],
    experiment_prefix="first-eval-in-langsmith",  # name of the experiment
)
