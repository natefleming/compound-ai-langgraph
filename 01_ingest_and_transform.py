# Databricks notebook source
# MAGIC %pip install --upgrade transformers pypdf langchain-text-splitters databricks-vectorsearch mlflow tiktoken torch llama-index docling
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

spark.conf.set("spark.sql.shuffle.partitions", 8)


# COMMAND ----------

from typing import Any, Dict, Optional, List

from mlflow.models import ModelConfig


model_config_file: str = "model_config.yaml"
model_config: ModelConfig = ModelConfig(development_config=model_config_file)

databricks_resources: Dict[str, Any] = model_config.get("databricks_resources")
retriever_config: Dict[str, Any] = model_config.get("retriever_config")
schema: Dict[str, Any] = retriever_config.get("schema")


catalog_name: str = databricks_resources.get("catalog_name")
database_name: str = databricks_resources.get("database_name")
volume_name: str = databricks_resources.get("volume_name")
source_path: str = databricks_resources.get("source_path")

users: List[str] = databricks_resources.get("users")
source_table_name: str = retriever_config.get("source_table_name")
vector_search_index: str = retriever_config.get("vector_search_index")
vector_search_endpoint_name: str = databricks_resources.get(
    "vector_search_endpoint_name"
)
primary_key: str = schema.get("primary_key")
embedding_source_column: str = schema.get("embedding_source_column")
embedding_model_endpoint_name: str = databricks_resources.get(
    "embedding_model_endpoint_name"
)

print(f"{catalog_name=}")
print(f"{database_name=}")
print(f"{volume_name=}")
print(f"{source_path=}")
print(f"{source_table_name=}")
print(f"{vector_search_endpoint_name=}")
print(f"{vector_search_index=}")
print(f"{primary_key=}")
print(f"{embedding_source_column=}")
print(f"{embedding_model_endpoint_name=}")
print(f"{users=}")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
  CatalogInfo, 
  SchemaInfo, 
  VolumeInfo, 
  VolumeType,
  SecurableType,
  PermissionsChange,
  Privilege
)

import app.catalog

w: WorkspaceClient = WorkspaceClient()

catalog: CatalogInfo 
try:
  catalog = w.catalogs.get(catalog_name)
except Exception as e: 
  catalog = w.catalogs.create(catalog_name)

schema: SchemaInfo
try:
  schema = w.schemas.get(f"{catalog.full_name}.{database_name}")
except Exception as e:
  schema = w.schemas.create(database_name, catalog.full_name)
  
volume: VolumeInfo
try:
  volume = w.volumes.read(f"{catalog.full_name}.{database_name}.{volume_name}")
except Exception as e:
  volume = w.volumes.create(catalog.full_name, schema.name, volume_name, VolumeType.MANAGED)

for user in users:
  user: str
  w.grants.update(
    full_name=catalog.full_name,
    securable_type=SecurableType.CATALOG,
    changes=[
      PermissionsChange(add=[Privilege.ALL_PRIVILEGES], principal=user)
    ])
  
spark.sql(f"USE {catalog.name}.{database_name}")

# COMMAND ----------

from pathlib import Path

#prgsupportdocuments.default.pandakb

raw_documents_path: Path = Path(source_path) #volume.as_path() / "sample_documents"
raw_checkpoint_path: Path = volume.as_path() / "checkpoints/raw_docs"

w.files.create_directory(raw_documents_path.as_posix())

print(f"{raw_documents_path.as_posix()=}")
print(f"{raw_checkpoint_path.as_posix()=}")



# COMMAND ----------

spark.sql(f"""
   DROP TABLE IF EXISTS {catalog_name}.{database_name}.raw_docs
""")

# COMMAND ----------

# MAGIC %sh rm -rf /Volumes/dbcks_poc/paws/data/checkpoints/raw_docs

# COMMAND ----------

from pyspark.sql import DataFrame

import pyspark.sql.functions as F
import pyspark.sql.types as T

from app.udf import guess_mime_type


df: DataFrame = (
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "BINARYFILE")
    .option("pathGlobFilter", "*.{pdf,docx,doc,ppt,pptx}")
    .load(raw_documents_path.as_posix())
)

df = df.withColumn("mime_type", guess_mime_type(df.path))
(
    df.writeStream.trigger(availableNow=True)
    .option("checkpointLocation", raw_checkpoint_path)
    .table(f"{catalog_name}.{database_name}.raw_docs")
    .awaitTermination()
)

# COMMAND ----------

display(spark.table(f"{catalog_name}.{database_name}.raw_docs"))

# COMMAND ----------

from typing import Iterator

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document, set_global_tokenizer

from transformers import AutoTokenizer

import pandas as pd

import pyspark.sql.functions as F

from app.udf import parse_bytes

# Reduce the arrow batch size as our PDF can be big in memory
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 10)

@F.pandas_udf("array<string>")
def read_as_chunk(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    # set llama2 as tokenizer to match our model size (will stay below gte 1024 limit)
    set_global_tokenizer(
        AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    )
    # Sentence splitter from llama_index to split on sentences
    splitter = SentenceSplitter(chunk_size=500, chunk_overlap=10)

    def extract_and_split(b):
        txt = parse_bytes(b)
        if txt is None:
            return []
        nodes = splitter.get_nodes_from_documents([Document(text=txt)])
        return [n.text for n in nodes]

    for x in batch_iter:
        yield x.apply(extract_and_split)

# COMMAND ----------

source_table_checkpoint_path: Path = volume.as_path() / "checkpoints/document_chunked"
print(source_table_checkpoint_path)

# COMMAND ----------

spark.sql(f"""
   DROP TABLE IF EXISTS {source_table_name}       
""")

# COMMAND ----------

# MAGIC %sh rm -rf /Volumes/dbcks_poc/paws/data/checkpoints/document_chunked

# COMMAND ----------


document_uri: str = "source"

spark.sql(f"""
  CREATE TABLE IF NOT EXISTS {source_table_name} (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY,
    {document_uri} STRING,
    content STRING
  ) TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

# COMMAND ----------

display(spark.table(source_table_name))

# COMMAND ----------

display(spark.read.table("raw_docs").select("mime_type").distinct())

# COMMAND ----------

#application/vnd.openxmlformats-officedocument.wordprocessingml.document
#application/vnd.openxmlformats-officedocument.presentationml.presentation
#application/pdf

# COMMAND ----------

from pyspark.sql import DataFrame

raw_docs_df: DataFrame = spark.read.table("raw_docs").limit(10)


raw_word_docs_df: DataFrame = spark.read.table("raw_docs").filter("mime_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'").limit(1)


raw_pdf_docs_df: DataFrame = spark.read.table("raw_docs").filter("mime_type = 'application/pdf'").limit(1)


raw_ppt_docs_df: DataFrame = spark.read.table("raw_docs").filter("mime_type = 'application/vnd.openxmlformats-officedocument.presentationml.presentation'").limit(1)


# COMMAND ----------

raw_word_docs_df.count()

# COMMAND ----------

df = (
    raw_docs_df
    .withColumn("content", F.explode(read_as_chunk("content")))
    .selectExpr(f"path as {document_uri}", "content")
)

display(df)


# COMMAND ----------

import pyspark.sql.functions as F

document_uri: str = "source"

(
    spark.readStream.table("raw_docs")
        .withColumn("content", F.explode(read_as_chunk("content")))
        .selectExpr(f"path as {document_uri}", "content")
        .writeStream.trigger(availableNow=True)
        .option("maxFilesPerTrigger", 5)
        .option("checkpointLocation", source_table_checkpoint_path)
        .table(source_table_name)
        .awaitTermination()
)

# COMMAND ----------

display(spark.table(source_table_name))
