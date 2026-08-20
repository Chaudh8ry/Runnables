from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

# Initialize the model
model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

# Output parser to convert model response into plain string
parser = StrOutputParser()

# Prompt template for short explanation
short_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in 1-2 Lines"
)

# Prompt template for detailed explanation
detailed_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in detail"
)

topic = "Parallel Runnables"

# -------------------------------
# Sequential way 
# -------------------------------
# formatted_short = short_prompt.format_messages(topic=topic)
# response = model.invoke(formatted_short)
# str_output = parser.parse(response.text)
#
# formatted_detail = detailed_prompt.format_messages(topic=topic)
# response = model.invoke(formatted_detail)
# str_output_detail = parser.parse(response.text)
#
# In sequential execution, you run one chain fully (prompt → model → parser),
# then repeat the same steps for the second chain. This means waiting for
# the first to finish before starting the second.

# -------------------------------
# Parallel way (RunnableParallel)
# -------------------------------
chain = RunnableParallel({
    # Define two parallel branches:
    # Each branch is a pipeline: prompt → model → parser
    "short": RunnableLambda(lambda x:x['short']) | short_prompt | model | parser,
    "detailed": RunnableLambda(lambda x:x['detailed']) | detailed_prompt | model | parser
})

# Invoke the parallel chain with input
result = chain.invoke({
    "short": {"topic": "parallel runnables"},
    "detailed": {"topic": "bm25 in RAG"}
})

# Each branch runs simultaneously and returns its own result
print(result['short'])     # Output from the short explanation branch
print(result['detailed'])  # Output from the detailed explanation branch
