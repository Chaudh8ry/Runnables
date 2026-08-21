from dotenv import load_dotenv
load_dotenv()

from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI

search_tool = TavilySearch(max_results = 5)

llm = ChatMistralAI(model="mistral-small-2506")

prompt = ChatPromptTemplate([
    ("system","""
    You are a helpful assistant
    Summarize the following news into clear bullet points
    {news}
    """)
])

# Built a chain: format theprompt -> send it to the model -> then parse the response
chain = prompt | llm | StrOutputParser()

# fetching the AI news
searchResult = search_tool.run("Ai news of 2026")

# passing the search result into chain
result = chain.invoke({"news": searchResult})

print(result)