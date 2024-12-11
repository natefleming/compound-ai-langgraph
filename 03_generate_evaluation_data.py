# Databricks notebook source
# MAGIC %pip install -U -qqqq databricks-agents mlflow databricks-sdk[openai] backoff
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %reload_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from typing import Any, Dict, Optional, List

from mlflow.models import ModelConfig


model_config_file: str = "model_config.yaml"
model_config: ModelConfig = ModelConfig(development_config=model_config_file)

databricks_resources: Dict[str, Any] = model_config.get("databricks_resources")
retriever_config: Dict[str, Any] = model_config.get("retriever_config")

evaluation_table_name: str = databricks_resources.get("evaluation_table_name")
source_table_name: str = retriever_config.get("source_table_name")

print(f"{evaluation_table_name=}")
print(f"{source_table_name=}")


# COMMAND ----------

from pyspark.sql import DataFrame
import pandas as pd

parsed_docs_df: DataFrame = spark.table(source_table_name).withColumnRenamed("url", "doc_uri")
parsed_docs_pdf: pd.DataFrame = parsed_docs_df.toPandas()

display(parsed_docs_pdf)

# COMMAND ----------

from pyspark.sql import DataFrame

# Use the synthetic eval generation API to get some evals
from databricks.agents.evals import generate_evals_df

# "Ghost text" for agent description and question guidelines - feel free to modify as you see fit.
agent_description = f"""
The agent is a RAG chatbot that answers questions about processes, troubleshooting, checklists and how-to guides. 
"""
question_guidelines = f"""
# User personas
- An employee who is looking for steps to troubleshoot a process or issue
- An experienced, highly technical Engineer

# Example questions
- How can I add a fundraiser key?
- How can I submit a referral bonus?

# Additional Guidelines
- Questions should be succinct, and human-like
"""

num_evals = 25
evals_pdf: pd.DataFrame = generate_evals_df(
    docs=parsed_docs_pdf[
        :500
    ],  # Pass your docs. They should be in a Pandas or Spark DataFrame with columns `content STRING` and `doc_uri STRING`.
    num_evals=num_evals,  # How many synthetic evaluations to generate
    agent_description=agent_description,
    question_guidelines=question_guidelines,
)

evals_df: DataFrame = spark.createDataFrame(evals_pdf)

evals_df.write.mode("overwrite").saveAsTable(evaluation_table_name)

