from typing import List

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

class Source(BaseModel):
    """Schema for a source used by the agent"""

    url:str = Field(description="the url of the source")

class AgentResponse(BaseModel):
    """Schema for the agent response with answer and sources"""

    answer:str = Field(description="the agents answer to the query")
    sources: list[Source] = Field(default_factory=list, description="list of sources used to generate the answer")


llm = ChatOpenAI(model="gpt-5")
# llm = ChatOllama(model="gemma4")
tools = [TavilySearch()]
agent= create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main() -> None:
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": [HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details")]})
    print(result)
