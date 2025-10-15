# from dotenv import load_dotenv
# load_dotenv()


# from langgraph.prebuilt import create_react_agent


# agent = create_react_agent(
#    model="openai:gpt-4o-mini", 
#    tools=[], 
#    prompt="You are a helpful assistant" 
# )


# # Run the agents
# response = agent.invoke(
#    {"messages": [{"role": "user", "content": "what are large language models"}]}
# )
# print(response)

# Add file and Add folder tools

import os
from dotenv import load_dotenv
load_dotenv()
from langgraph.prebuilt import create_react_agent
def addFile(filename: str) -> str:
   """Create a new file in current directory"""
   if not os.path.exists(filename):
      with open(filename, "w") as f:
          pass
      print(f"File '{filename}' created.")
   else:
      print(f"File '{filename}' already exists.")

def addFolder(directory_name: str):
  """Create a new Directory in current directory"""
  if not os.path.exists(directory_name):
      os.mkdir(directory_name)
      print(f"Directory '{directory_name}' created.")
  else:
      print(f"Directory '{directory_name}' already exists.")

agent = create_react_agent(
  model="openai:gpt-4o-mini",
  tools=[addFile,addFolder],
  prompt=""
)
# Run the agents
response = agent.invoke(
  {"messages": [{"role": "user", "content": "create a file named test.txt in folder named dharma"}]}
)
print(response)