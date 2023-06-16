from github import Github
import random
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from langchain.llms import openai
import langchain


# Set up OpenAI API
openai.api_key = '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'  # Replace with your OpenAI API key

# Set up LangChain
langchain.setup('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')  # Replace with your LangChain API key

# Authenticate using a personal access token
access_token = "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"  # Replace with your GitHub access token
g = Github(access_token)

# Prompt user for GitHub username
username = input("Enter GitHub username: ")

class YourLLMClass(langchain.LLModel):
    def __init__(self, model_name):
        super().__init__(model_name)

    def predict(self, prompt):
        # Implement the predict method to generate LLM predictions for a given prompt
        # Use LangChain's functionality to assess the technical complexity or any other criteria

        # Example implementation:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=50
        )
        return response.choices[0].text.strip()

try:
    # Get the user
    user = g.get_user(username)

    # Fetch repositories
    repositories = user.get_repos()

    # Initialize your LLM class
    llm = YourLLMClass(model_name="gpt2")  # Replace with appropriate configuration

    def preprocess_code(code):
        # Split code into chunks
        code_chunks = code.split("\n\n")  # Split by double line breaks

        # Split large files into smaller chunks
        MAX_CHUNK_SIZE = 500  # Set maximum chunk size
        chunked_code = []
        for chunk in code_chunks:
            if len(chunk) <= MAX_CHUNK_SIZE:
                chunked_code.append(chunk)
            else:
                # Split chunk into smaller sub-chunks
                sub_chunks = [chunk[i:i + MAX_CHUNK_SIZE] for i in range(0, len(chunk), MAX_CHUNK_SIZE)]
                chunked_code.extend(sub_chunks)

        # Randomly sample representative parts of the code
        MAX_SAMPLES = 3  # Set maximum number of code samples
        sampled_code = random.sample(chunked_code, min(len(chunked_code), MAX_SAMPLES))

        return sampled_code

    def evaluate_complexity_with_gpt(code):
        # Initialize GPT-2 model and tokenizer
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        # Define the prompt for evaluating complexity
        prompt = "Evaluate the technical complexity of the following code:\n\n" + code

        # Tokenize the prompt
        input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors='pt')

        # Generate the complexity score using GPT-2
        with torch.no_grad():
            gpt_outputs = model.generate(input_ids, max_length=20, num_return_sequences=1)

        gpt_score = gpt_outputs[0]['generated_text'].split()[-1]

        return float(gpt_score)

    def evaluate_complexity_with_langchain(code):
        # Use LangChain to evaluate the complexity of the code
        langchain_score = llm.predict(complexity_prompt + code)

        return float(langchain_score)

    # Define prompts for technical complexity evaluation
    complexity_prompt = "Evaluate the technical complexity of the code."

    # Variables to track the most complex repository and its complexity analysis
    max_complexity = float('-inf')
    most_complex_repo = None

    # Process the fetched repositories
    for repo in repositories:
        repo_name = repo.name
        repo_url = repo.html_url
        repo_description = repo.description

        # Print repository information
        print("Repository Name:", repo_name)
        print("Repository URL:", repo_url)
        print("Repository Description:", repo_description)

        # Fetch repository contents
        contents = repo.get_contents("")

        # Process each file in the repository
        for content in contents:
            if content.type == "file":
                # Fetch file code
                code = content.decoded_content.decode()

                # Preprocess the code
                preprocessed_code = preprocess_code(code)

                # Evaluate technical complexity using GPT-2 and LangChain
                complexity_scores = []
                for code_chunk in preprocessed_code:
                    # Assess complexity using GPT-2
                    gpt_score = evaluate_complexity_with_gpt(code_chunk)

                    # Assess complexity using LangChain
                    langchain_score = evaluate_complexity_with_langchain(code_chunk)

                    # Update the complexity scores
                    complexity_scores.append((gpt_score, langchain_score))

                # Calculate the average complexity score for the code chunk
                avg_gpt_score = sum(gpt_score for gpt_score, _ in complexity_scores) / len(complexity_scores)
                avg_langchain_score = sum(langchain_score for _, langchain_score in complexity_scores) / len(complexity_scores)

                # Update the most complex repository if needed
                if avg_gpt_score > max_complexity and avg_langchain_score > max_complexity:
                    max_complexity = max(avg_gpt_score, avg_langchain_score)
                    most_complex_repo = repo

    # Print the most complex repository
    if most_complex_repo:
        print("\nMost Complex Repository:")
        print("Repository Name:", most_complex_repo.name)
        print("Repository URL:", most_complex_repo.html_url)
        print("Repository Description:", most_complex_repo.description)
        print("Complexity Score:", max_complexity)
    else:
        print("No repositories found for the given username.")

except Exception as e:
    print("Error occurred:", str(e))
