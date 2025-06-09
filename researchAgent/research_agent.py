import os
from typing import Type

from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

os.environ['SERPER_API_KEY'] = SERPER_API_KEY
os.environ['GROQ_API_KEY'] = GROQ_API_KEY

class PatchedSchema(BaseModel):
    search_query: str

    @classmethod
    def validate(cls, value):
        # Handle direct dict passed in
        if isinstance(value, dict):
            if "search_query" in value and isinstance(value["search_query"], dict):
                value["search_query"] = value["search_query"].get("description") or str(value["search_query"])
        return super().validate(value)

# Step 2: Create patched tool with this schema
class FullyPatchedSerperDevTool(SerperDevTool):
    args_schema: Type[BaseModel] = PatchedSchema

    def run(self, query):
        # Defensive parsing
        if isinstance(query, dict):
            query = query.get("search_query") or query.get("description") or str(query)
        return super()._run(search_query=query)
tool = FullyPatchedSerperDevTool()

def create_research_agent():
    llm = ChatGroq(
        model="groq/llama-3.1-8b-instant",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )
    return Agent(
        role="Research Agent",
        goal="Conduct thorough research on given topics",
        backstory="You are an experienced researcher with expertise in finding and synthesizing information from various sources",
        verbose=True,
        allow_delegation=False,
        tools=[tool],
        llm=llm
    )
    

def create_task(agent,topic):
    return Task(
        description=f"Research the following topic and provide a comprehensive summary: {topic}",
        agent=agent,
        expected_output = "A detailed summary of the research findings, including key points and insights related to the topic"
    )
   

def run_research(topic):
    agent = create_research_agent()
    task = create_task(agent,topic)
    crew = Crew(
        agents=[agent], 
        tasks=[task],
        verbose=True,
        process=Process.sequential,
    )
    print(crew)
    result = crew.kickoff()
    return result

if __name__ == "__main__":
    topic = input("Enter the topic you want to research: ")
    result = run_research(topic)
    print(result)