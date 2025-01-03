# Databricks notebook source
# MAGIC %pip install -U -qqqq databricks-agents mlflow databricks-sdk[openai] backoff openpyxl
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
curated_evaluation_table_name: str = databricks_resources.get("curated_evaluation_table_name")
source_table_name: str = retriever_config.get("source_table_name")

print(f"{evaluation_table_name=}")
print(f"{curated_evaluation_table_name=}")
print(f"{source_table_name=}")


# COMMAND ----------

from pyspark.sql import DataFrame
import pandas as pd

parsed_docs_df: DataFrame = spark.table(source_table_name).withColumnRenamed("source", "doc_uri")
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
- Questions should be succinct and human-like
- Questions should not contain any sensitive information
- Only use questions that are relevant to the domain
- Omit questions that are too broad or vague
- Omit questions that do not have a clear answer
"""

num_evals: int = 20
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


# COMMAND ----------

display(spark.table(evaluation_table_name))

# COMMAND ----------

from pyspark.sql import DataFrame
import pyspark.sql.functions as F

evaluation_df: DataFrame = spark.table(evaluation_table_name)
messages_df: DataFrame = evaluation_df.select("request_id", F.explode("request.messages").alias("messages"))
questions_df: DataFrame = messages_df.select("request_id", "messages.content")

display(questions_df)

# COMMAND ----------

questions_df = (
    questions_df.where(
        (questions_df.content.isNull()) | 
        (F.length(questions_df.content) <= 20)
    )
)
display(questions_df)

# COMMAND ----------

invalid_question_ids = [r.request_id for r in questions_df.select("request_id").distinct().collect()]
(
    spark.table(evaluation_table_name)
    .where(~F.col("request_id").isin(invalid_question_ids))
    .write
    .mode("overwrite")
    .saveAsTable(evaluation_table_name)
)


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Create Dataset from KB FAQ and Expected Answers

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.files import DirectoryEntry

from pyspark.sql import DataFrame
import pandas as pd

w: WorkspaceClient = WorkspaceClient()


evaluation_path: str = "/Volumes/dbcks_poc/paws/data/evaluation"

evaluation_pdf: pd.DataFrame = pd.DataFrame()

for entry in w.files.list_directory_contents(evaluation_path):
    entry: DirectoryEntry
    path: str = entry.path
    if path.endswith(".xlsx"):
        evaluation_pdf = pd.concat([evaluation_pdf, pd.read_excel(path)], ignore_index=True)

evaluation_df: DataFrame = spark.createDataFrame(evaluation_pdf)

display(evaluation_df)

# COMMAND ----------

from typing import Dict, List

import pyspark.sql.functions as F
import pyspark.sql.types as T

# ChatMessageSchema: T.StructType = T.StructType([
#     T.StructField("role", T.StringType(), False),
#     T.StructField("content", T.StringType(), False),
# ])

# ChatCompletionSchema = T.StructType([
#     T.StructField("messages", T.ArrayType(ChatMessageSchema), False),
# ])


# @F.udf(returnType=ChatCompletionSchema)
# def as_chat_completion(question: str) -> ChatCompletionSchema:
#     messages = {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": question
#             },
#         ]
#     }
#     return messages


# chat_completion_df: DataFrame = (
#     evaluation_df.withColumns(
#         { 
#          "request_id" : F.expr("uuid()"),
#          "request": as_chat_completion(F.col("Question")) 
#         }
#     )
# )


ExpectedRetrievedContextSchema = T.ArrayType(
    T.StructType([
        T.StructField("content", T.StringType(), False),
        T.StructField("doc_uri", T.StringType(), False),
    ])
)

chat_completion_df: DataFrame = (
    evaluation_df.withColumns(
        {
            "request_id": F.expr("uuid()"),
            "request": F.col("Question"),
            "expected_response": F.col("Expected Answer"),
            "expected_retrieved_context": F.lit([]).cast(ExpectedRetrievedContextSchema),
        }
    ).select("request_id", "request", "expected_response", "expected_retrieved_context")
)
display(chat_completion_df)

chat_completion_df.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(curated_evaluation_table_name)
