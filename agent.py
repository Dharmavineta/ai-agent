import os
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process

llm = ChatGoogleGenerativeAI(model="gemini-pro",verbose = True,temperature = 0.6,google_api_key="Aiekmasskeifhasskennvaldkeiahdklakn")

from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

researcher = Agent(
  role='Researcher',
  goal='You provide results on the basis of Facts and only Facts along with supported doc related urls,You go to the root cause and give the best possible outcomes',
  backstory="You're an job research agent, you'll search for jobs available in market and those which are open now, avoid showing closed jobs",
  verbose=True,
  allow_delegation=False,
  llm = llm,  
  tools=[search_tool]
)

task1 = Task(
  description="I'm looking for react based openings",
  agent=researcher,
  expected_output="give me clear and concise react openings"
)

crew = Crew(
  agents=[researcher],
  tasks=[task1],
  verbose=2, 
  process=Process.sequential
)

result = crew.kickoff()

print("######################")
print(result)