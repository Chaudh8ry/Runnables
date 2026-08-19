# A RunnableSequence is essentially a pipeline: it takes input, passes it through a sequence of steps (prompt → model → parser), and returns the final output.
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
# Import prompt template utilities
from langchain_core.prompts import ChatPromptTemplate
# Import output parser to convert model responses into plain strings
from langchain_core.output_parsers import StrOutputParser

# -------------------------------
# Step 1: Define a prompt template
# -------------------------------
# This template will accept a variable {topic} and generate a message
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words"
)

# -------------------------------
# Step 2: Initialize the model
# -------------------------------
# Here we specify which Gemini variant to use
model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

# -------------------------------
# Step 3: Define an output parser
# -------------------------------
# This parser ensures the raw model output is converted into a clean string
parser = StrOutputParser()

# -------------------------------
# OLD WAY: Manual step-by-step flow
# -------------------------------
# Format the prompt with a specific topic
formatted_prompt = prompt.format_messages(topic="Machine Learning")

# Send the formatted prompt to the model
response = model.invoke(formatted_prompt)

# Parse the raw response text into final output
final_output = parser.parse(response.text)

# -------------------------------
# NEW WAY: Chain composition
# -------------------------------
# Instead of manually wiring prompt → model → parser,
# LangChain lets us "pipe" them together into a single chain.
chain = prompt | model | parser

# Now we can directly invoke the chain with just the input variable.
result = chain.invoke("Inference Engineering")

# Print the final result
print(result)
