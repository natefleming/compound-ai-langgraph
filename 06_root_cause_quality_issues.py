# Databricks notebook source
# MAGIC %pip install -U -qqqq databricks-agents mlflow mlflow-skinny databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from databricks import agents
import pandas as pd

# COMMAND ----------

# MAGIC %md # Load the evaluation data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get the MLflow run of the POC application 

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

MLFLOW_EXPERIMENT_NAME: str = "/Users/nate.fleming@databricks.com/compound-ai-langgraph/04_chain_as_code_driver"
POC_CHAIN_RUN_NAME: str = "chain"

# COMMAND ----------

runs = mlflow.search_runs(experiment_names=[MLFLOW_EXPERIMENT_NAME], filter_string=f"run_id = '378aa79cb080476287437e8aa6012128'", output_format="list")

if len(runs) != 1:
    raise ValueError(f"Found {len(runs)} run with name {POC_CHAIN_RUN_NAME}.  Ensure the run name is accurate and try again.")

poc_run = runs[0]

# COMMAND ----------

poc_run

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the evaluation data

# COMMAND ----------

mc = mlflow.MlflowClient()
eval_results_df = mc.load_table(experiment_id=poc_run.info.experiment_id, run_ids=[poc_run.info.run_id], artifact_file="eval_results.json")
display(eval_results_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Root cause analysis
# MAGIC
# MAGIC Below you will see a few examples of how to link the evaluation results back to potential root causes and their corresponding fixes

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example: Find requests that are incorrect and have low retrieval accuracy. Potential fix: improve the retriever

# COMMAND ----------

#https://docs.databricks.com/en/generative-ai/agent-evaluation/evaluation-schema.html

# display(eval_results_df[(eval_results_df["response/llm_judged/correctness/rating"]=="no") & (eval_results_df["retrieval/llm_judged/chunk_relevance_precision"]<.5)])

#retrieval/ground_truth/document_recall/average

display(eval_results_df[(eval_results_df["response/llm_judged/correctness/rating"]=="no")])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example: Find requests that are not grounded even though retrieval accuracy is high. Potential fix: tune the generator prompt to avoid hallucinations.

# COMMAND ----------

# display(eval_results_df[(eval_results_df["response/llm_judged/groundedness/rating"]=="no") & (eval_results_df["retrieval/llm_judged/chunk_relevance/precision"]>=.5)])

display(eval_results_df[(eval_results_df["response/llm_judged/groundedness/rating"]=="no")])

