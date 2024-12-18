from dotenv import load_dotenv
import os
import dspy
from dspy.teleprompt import LM

# Load environment variables from .env file
load_dotenv()
# Initialize OpenAI client with API key from environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')


# Configure DSPy with OpenAI
lm = LM(model_name="gpt-4", api_key=openai_api_key)
dspy.configure(lm=lm)

qa = dspy.Predict('question: str -> response: str')
response = qa(question="what are high memory and low memory on linux?")

print(response.response)