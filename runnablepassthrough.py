from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnablePassthrough

model = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite')
parser = StrOutputParser()

code_prompt = ChatPromptTemplate([
    ("system","You are a code generator"),
    ("human","{topic}")
])

explain_prompt = ChatPromptTemplate([
    ("system","Your are a helpful assistant who explains code in simple terms"),
    ("human","Explain the following code in simple words: \n {code}")
])

seq1 = code_prompt | model | parser

seq2 = RunnableParallel(
    {
        "code_generated" : RunnablePassthrough(),
        "explanation" :  explain_prompt | model | parser
    }
)

chain = seq1 | seq2
result = chain.invoke({"topic": "Code for linked list in python"})
print(result['code_generated'])
print(result['explanation'])