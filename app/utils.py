from databricks.vector_search.client import VectorSearchClient

def get_latest_model_version(model_name: str) -> int:
    """Get the latest version of a model from the MLflow model registry.

    Args:
        model_name (str): The name of the model.

    Returns:
        int: The latest version number of the model.
    """
    from mlflow.tracking import MlflowClient
    mlflow_client: MlflowClient = MlflowClient(registry_uri="databricks-uc")
    latest_version: int = 1
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

def endpoint_exists(vsc: VectorSearchClient, vs_endpoint_name: str) -> bool:
    """Check if a vector search endpoint exists.

    Args:
        vsc (VectorSearchClient): The vector search client.
        vs_endpoint_name (str): The name of the vector search endpoint.

    Returns:
        bool: True if the endpoint exists, False otherwise.
    """
    try:
        return vs_endpoint_name in [e['name'] for e in vsc.list_endpoints().get('endpoints', [])]
    except Exception as e:
        if "REQUEST_LIMIT_EXCEEDED" in str(e):
            print("WARN: couldn't get endpoint status due to REQUEST_LIMIT_EXCEEDED error.")
            return True
        else:
            raise e


def wait_for_vs_endpoint_to_be_ready(vsc: VectorSearchClient, vs_endpoint_name: str) -> None:
    """Wait for a vector search endpoint to be ready.

    Args:
        vsc (VectorSearchClient): The vector search client.
        vs_endpoint_name (str): The name of the vector search endpoint.

    Raises:
        Exception: If the endpoint is not ready within the timeout period.
    """
    for i in range(180):
        try:
            endpoint = vsc.get_endpoint(vs_endpoint_name)
        except Exception as e:
            if "REQUEST_LIMIT_EXCEEDED" in str(e):
                print("WARN: couldn't get endpoint status due to REQUEST_LIMIT_EXCEEDED error. Please manually check your endpoint status")
                return
            else:
                raise e
        status = endpoint.get("endpoint_status", endpoint.get("status"))["state"].upper()
        if "ONLINE" in status:
            return endpoint
        elif "PROVISIONING" in status or i < 6:
            if i % 20 == 0:
                print(f"Waiting for endpoint to be ready, this can take a few min... {endpoint}")
            time.sleep(10)
        else:
            raise Exception(f'''Error with the endpoint {vs_endpoint_name}. - this shouldn't happen: {endpoint}.\n Please delete it and re-run the previous cell: vsc.delete_endpoint("{vs_endpoint_name}")''')
    raise Exception(f"Timeout, your endpoint isn't ready yet: {vsc.get_endpoint(vs_endpoint_name)}")


def index_exists(vsc: VectorSearchClient, endpoint_name: str, index_full_name: str) -> bool:
    """Check if an index exists in a vector search endpoint.

    Args:
        vsc (VectorSearchClient): The vector search client.
        endpoint_name (str): The name of the vector search endpoint.
        index_full_name (str): The full name of the index.

    Returns:
        bool: True if the index exists, False otherwise.
    """
    try:
        vsc.get_index(endpoint_name, index_full_name).describe()
        return True
    except Exception as e:
        if 'RESOURCE_DOES_NOT_EXIST' not in str(e):
            print(f'Unexpected error describing the index. This could be a permission issue.')
            raise e
    return False


def wait_for_index_to_be_ready(vsc: VectorSearchClient, vs_endpoint_name: str, index_name: str) -> None:
    """Wait for an index to be ready in a vector search endpoint.

    Args:
        vsc (VectorSearchClient): The vector search client.
        vs_endpoint_name (str): The name of the vector search endpoint.
        index_name (str): The name of the index.

    Raises:
        Exception: If the index is not ready within the timeout period.
    """
    for i in range(180):
        idx = vsc.get_index(vs_endpoint_name, index_name).describe()
        index_status = idx.get('status', idx.get('index_status', {}))
        status = index_status.get('detailed_state', index_status.get('status', 'UNKNOWN')).upper()
        url = index_status.get('index_url', index_status.get('url', 'UNKNOWN'))
        if "ONLINE" in status:
            return
        if "UNKNOWN" in status:
            print(f"Can't get the status - will assume index is ready {idx} - url: {url}")
            return
        elif "PROVISIONING" in status:
            if i % 40 == 0:
                print(f"Waiting for index to be ready, this can take a few min... {index_status} - pipeline url:{url}")
            time.sleep(10)
        else:
            raise Exception(f'''Error with the index - this shouldn't happen. DLT pipeline might have been killed.\n Please delete it and re-run the previous cell: vsc.delete_index("{index_name}, {vs_endpoint_name}") \nIndex details: {idx}''')
    raise Exception(f"Timeout, your index isn't ready yet: {vsc.get_index(index_name, vs_endpoint_name)}")