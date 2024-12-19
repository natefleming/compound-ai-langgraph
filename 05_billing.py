# Databricks notebook source
# MAGIC %md
# MAGIC Model Serving Endpoint

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT SUM(usage_quantity) as dbus 
# MAGIC from system.billing.usage 
# MAGIC where billing_origin_product = 'MODEL_SERVING' and usage_metadata.endpoint_name = 'agents_pandas_poc-paws-compound_ai_langgraph'

# COMMAND ----------

# MAGIC %md 
# MAGIC Agent Evaluation DBUs

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT SUM(usage_quantity) as dbus 
# MAGIC from system.billing.usage 
# MAGIC where billing_origin_product = 'AGENT_EVALUATION' 

# COMMAND ----------

# MAGIC %md
# MAGIC Daily DBUs

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC WITH all_vector_search_usage (
# MAGIC   SELECT *,
# MAGIC          CASE WHEN usage_metadata.endpoint_name IS NULL
# MAGIC               THEN 'ingest'
# MAGIC               ELSE 'serving'
# MAGIC         END as workload_type
# MAGIC     FROM system.billing.usage
# MAGIC    WHERE billing_origin_product = 'VECTOR_SEARCH'
# MAGIC ),
# MAGIC daily_dbus AS (
# MAGIC   SELECT workspace_id,
# MAGIC        cloud,
# MAGIC        usage_date,
# MAGIC        workload_type,
# MAGIC        usage_metadata.endpoint_name as vector_search_endpoint,
# MAGIC        SUM(usage_quantity) as dbus
# MAGIC  FROM all_vector_search_usage
# MAGIC  GROUP BY all
# MAGIC ORDER BY 1,2,3,4,5 DESC
# MAGIC )
# MAGIC SELECT * FROM daily_dbus

# COMMAND ----------

# MAGIC %sql
# MAGIC select
# MAGIC   usage_metadata.endpoint_name as endpoint_name, usage_quantity as dbus, sku_name
# MAGIC from
# MAGIC   system.billing.usage
# MAGIC where
# MAGIC   billing_origin_product = 'VECTOR_SEARCH'
# MAGIC   and usage_metadata.endpoint_name is not null 
# MAGIC   and usage_date between date_add(current_date(), -30) and current_date()
# MAGIC   and sku_name like "%_SERVERLESS_REAL_TIME_INFERENCE_%" 
