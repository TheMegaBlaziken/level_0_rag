import weaviate
from weaviate.connect import ConnectionParams

def delete_all_collections():
    client = weaviate.WeaviateClient(
        connection_params=ConnectionParams.from_url("http://localhost:8080", grpc_port=50051)
    )
    client.connect()
    for collection in client.collections.list_all():
        print(f"[INFO] Deleting all collections from the Weaviate cluster...")
        client.collections.delete(collection)
    print("[INFO] All collections deleted.")
    client.close()

if __name__ == "__main__":
    delete_all_collections()