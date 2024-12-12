
from typing import Any, Dict, List, Optional

from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_databricks.vectorstores import DatabricksVectorSearch

import mlflow.models


from langchain.schema import Document
from typing import List


class MetadataMapperRetriever(VectorStoreRetriever):

    def __init__(self, target: VectorStoreRetriever) -> None:
        #super().__init__()  # Initialize base class
        self.target = target

    def get_relevant_documents(self, query: str) -> List[Document]:
        documents = self.target.get_relevant_documents(query)
        for doc in documents:
            if "url" in doc.metadata:
                doc.metadata["source"] = doc.metadata["url"]  # Map 'url' to 'source'
        return documents

    def add_document(self, *args, **kwargs):
        return self.target.add_document(*args, **kwargs)

    def close(self):
        self.target.close()


def create_vector_search_retriever(
    endpoint_name: str,
    index_name: str,
    primary_key: Optional[str] = None,
    text_column: Optional[str] = None,
    doc_uri: Optional[str] = None,
    columns: Optional[List[str]] = None,
    parameters: Optional[Dict[str, Any]] = None,
) -> VectorStoreRetriever:

    vector_search: DatabricksVectorSearch = DatabricksVectorSearch(
        index_name=index_name,
        endpoint=endpoint_name,
        text_column=text_column,
        columns=columns,
    )
    
    vector_search_as_retriever: VectorStoreRetriever = (
        vector_search.as_retriever(search_kwargs=parameters)
    )

    retriever: MetadataMapperRetriever = MetadataMapperRetriever(target=vector_search_as_retriever)

    mlflow.models.set_retriever_schema(
        name=index_name,
        primary_key=primary_key,
        text_column=text_column,
        doc_uri=doc_uri,
    )
    
    return retriever
