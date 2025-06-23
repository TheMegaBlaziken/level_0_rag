# clear_schema.py
import os
from weaviate import Client
from weaviate.auth import AuthApiKey

# grab your endpoint and API key from the env
WEAVIATE_URL     = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

# wrap your key in the AuthApiKey helper
auth = AuthApiKey(api_key=WEAVIATE_API_KEY)

# initialize the client
client = Client(
    url=WEAVIATE_URL,
    auth_client_secret=auth
)

# now drop the class and all its data
client.schema.delete_all()
