# Compound AI Using Langraph

* Databricks Genie Agent
* Databricks Vector Search Agent

## Usage

* Use the scratchpad notebook for interactive development
* Use chain_as_code_driver.py for model registry and model serving deployment
* **NOTE:** You will need to update Vector Search endpoint and index variables

## Extension

* Subclass the app.agents.AgentBase abstract class
* Register the new agent with a Graph using app.graph.GraphBuilder.add_agent(...)
  
