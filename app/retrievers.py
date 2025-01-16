
from typing import Any, Dict, List, Optional

from langchain_core.vectorstores.base import VectorStoreRetriever
from databricks_langchain.vectorstores import DatabricksVectorSearch

import mlflow.models


from langchain.schema import Document
from typing import List
import os

def create_vector_search_retriever(
    endpoint_name: str,
    index_name: str,
    primary_key: Optional[str] = None,
    text_column: Optional[str] = None,
    doc_uri: Optional[str] = None,
    columns: Optional[List[str]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    workspace_url: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
) -> VectorStoreRetriever:

    vector_search: DatabricksVectorSearch = DatabricksVectorSearch(
        index_name=index_name,
        endpoint=endpoint_name,
        text_column=text_column,
        columns=columns,
        client_args={
            "workspace_url": workspace_url,
            "service_principal_client_id": client_id,
            "service_principal_client_secret": client_secret,
        }
    )
    
    vector_search_as_retriever: VectorStoreRetriever = (
        vector_search.as_retriever(search_kwargs=parameters)
    )

    mlflow.models.set_retriever_schema(
        name=index_name,
        primary_key=primary_key,
        text_column=text_column,
        doc_uri=doc_uri,
    )
    
    return vector_search_as_retriever
