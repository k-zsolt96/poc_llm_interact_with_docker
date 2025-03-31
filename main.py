from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.llms import VLLMOpenAI
from langchain_ollama import ChatOllama
import json
import docker

#----PART 1: LLM INITIALIZATION----

#----PART 1-A: Ollama----
# Initialize the LLM - in this case using Ollama as the LLM backend
llm = ChatOllama(
    model="llama3.2", # name of the model to use, make sure to have it downloaded (ollama pull llama3.2)
    temperature=0, # temperature for the model, 0 for deterministic output
    format="json", # format of the output, json for structured output
    cache=False # whether to cache the response or not
)

#----PART 1-B: VLLM OpenAI API----
# VLLM OpenAI API client - example from https://python.langchain.com/docs/integrations/llms/vllm/#openai-compatible-completion
# Initialize the LLM - instead of ChatOllama from the above
# llm = VLLMOpenAI(
#     openai_api_key="EMPTY",
#     openai_api_base="http://localhost:8000/v1",
#     model_name="tiiuae/falcon-7b",
#     model_kwargs={"stop": ["."]},
# )

#----PART 2: LLM PROMPTING----

#----PART 2-A: Defining system/user prompts----
# An example of a system prompt to set the context for the LLM - uncomment PART 2-C to try this, PART 2-B does NOT use this
system_prompt = "You are an expert in writing CLI commands for Linux."

# An example of a user prompt
prompt = "Give me a commands that creates a hello.txt file in the /tmp folder. " \
"Your response should be a json with a single key 'command' and the value should be the command I asked."

#----PART 2-B: Invoking the LLM with the user prompt only----
# Invoke the LLM with the prompt
llm_response = llm.invoke(prompt)

#----PART 2-C: Invoking the LLM with both system and user prompts----
# This is an alternative way to invoke the LLM with a system message and a human message insted of a single prompt 
#
# llm_response = llm.invoke(
#             [SystemMessage(content=system_prompt)]
#             + [HumanMessage(content=prompt)],
#         )

#----PART 2-D: Parse the response from the LLM----
# Parse the response from the LLM into a JSON object
json_response = json.loads(llm_response.content)

# Extract the command from the JSON response
command = json_response["command"]

#----PART 2-E: Check the response from the LLM----
print(llm_response)
print()
print(json_response)
print()
print(command)

#----PART 3: RUNNING THE COMMAND IN A DOCKER CONTAINER----

#----Part 3-A: Create the docker container----
# Initialize Docker client
docker_client = docker.from_env()

# Name of the Docker image to use
image_name = "ubuntu:latest"

print(f"Pulling image {image_name}...")
# Pull the Docker image
image = docker_client.images.pull(image_name)

if not image:
    print(f"Failed to pull image {image_name}.")
# Create and start a Docker container from the previously pulled image
container = docker_client.containers.run(
    image=image_name,
    command="sleep infinity",
    detach=True,
    tty=True,
)

#---- Part 3-B: Run the command from the LLM inside the docker container----
# Run the command from the LLM in the Docker container
exit_code, output = container.exec_run(command)

if exit_code:
    print(f"Command encounted an error and exited with exit code {exit_code}.")

#----PART 4: FEEDBACK TO THE LLM----

#----PART 4-A: Create the feedback prompt to the LLM----
# Another example of a user prompt with feedback from the command execution    
prompt2 = f"By running the following command: `{command}`, I receive the following exit code: `{exit_code}` and output: `{output}`. Please explain giving me a JSON output, with a single key 'explanation' and the value should be the explanation of the command and its output."

#----PART 4-B: Invoking the LLM with the feedback prompt----
# Invoke the LLM with the feedback
llm_response2 = llm.invoke(prompt2)

#----PART 4-C: Invoking the LLM with both system and user prompts----
# This is an alternative way to invoke the LLM with a system message and a human message insted of a single prompt
# NOTE: this example uses the same system prompt as above, but you can use a different one if you want 
#
# llm_response2 = llm.invoke(
#             [SystemMessage(content=system_prompt)]
#             + [HumanMessage(content=prompt2)],
#         )

#----PART 4-D: Parse the response from the LLM----
# Parse the response from the LLM into a JSON object
json_response2 = json.loads(llm_response2.content)

# Extract the command from the JSON response
explanation = json_response2["explanation"]

#----PART 4-E: Check the response from the LLM----
print(llm_response2)
print()
print(json_response2)
print()
print(explanation)
