import os
import threading
import logging
from configparser import ConfigParser
from dotenv import load_dotenv
import weaviate

from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes.query import MetadataQuery
from weaviate.classes.init import Auth
from weaviate.exceptions import (
    WeaviateConnectionError,
    ObjectAlreadyExistsError,
    WeaviateQueryError,
)

from rest_framework.response import Response
from rest_framework import status


load_dotenv()
cfg = ConfigParser()
cfg.read("conf/application.ini")

MAX_RESULTS = int(cfg["Weaviate"]["retrieval_input_max_rows"])
WEAVIATE_ENDPOINT = cfg["Weaviate"]["host"]
MODEL_NAME = cfg["Weaviate"]["model"]
VECTOR_SIZE = int(cfg["Weaviate"]["dimensions"])
API_TOKEN = os.getenv("WEAVIATE_API_KEY")

log = logging.getLogger("weaviate_service")


class WeaviateHandler:
    _singleton = None
    _thread_guard = threading.Lock()

    def __new__(cls):
        if cls._singleton is None:
            with cls._thread_guard:
                if cls._singleton is None:
                    cls._singleton = super(WeaviateHandler, cls).__new__(cls)
        return cls._singleton

    def __init__(self):
        if not hasattr(self, "_booted"):
            self._client = self._connect_client()
            self._booted = True


    def _connect_client(self):
        try:
            client = weaviate.connect_to_local(
                host=WEAVIATE_ENDPOINT,
                auth_credentials=Auth.api_key(API_TOKEN)
            )
            assert client.is_ready()
            return client
        except Exception:
            log.critical("Weaviate connection failed", exc_info=True)
            raise


    def _make_collection_name(self, org: str, is_image: bool = False) -> str:
        prefix = org.strip().title().replace(" ", "_")
        suffix = "_product_image" if is_image else "_product_description"
        return f"{prefix}{suffix}"

    def _define_schema(self, for_images: bool = False):
        base = [
            Property(name="url", data_type=DataType.TEXT, index_searchable=False, index_filterable=True),
            Property(name="name", data_type=DataType.TEXT, index_searchable=False, index_filterable=True),
        ]
        if for_images:
            base.append(
                Property(
                    name="fashion_descriptor",
                    data_type=DataType.TEXT,
                    index_searchable=True,
                    index_filterable=False,
                )
            )
        return base


    def shutdown(self):
        if getattr(self, "_client", None):
            self._client.close()
            self._client = None


    def create_collection(self, org_name: str):
        try:
            for img in [False, True]:
                name = self._make_collection_name(org_name, img)
                self._client.collections.create(
                    name,
                    properties=self._define_schema(img),
                    vectorizer_config=Configure.Vectorizer.text2vec_openai(
                        model=MODEL_NAME,
                        dimensions=VECTOR_SIZE,
                        type_="text",
                    ),
                    vector_index_config=Configure.VectorIndex.hnsw(
                        distance_metric=VectorDistances.COSINE
                    ),
                    sharding_config=Configure.sharding(
                        virtual_per_physical=128,
                        desired_count=1,
                        desired_virtual_count=128,
                    ),
                    replication_config=Configure.replication(factor=1),
                )
        except (WeaviateConnectionError, ObjectAlreadyExistsError) as e:
            log.critical(f"collection_creation_failed - [{org_name}]", exc_info=True)
            raise e
        except Exception:
            log.critical(f"unexpected_error_during_collection_creation - [{org_name}]", exc_info=True)
            raise

    def remove_collection(self, org_name: str):
        try:
            for img in [False, True]:
                name = self._make_collection_name(org_name, img)
                self._client.collections.delete(name)
        except WeaviateConnectionError as e:
            log.critical(f"weaviate_connection_failed - [{org_name}]", exc_info=True)
            raise e
        except Exception:
            log.critical(f"collection_removal_failed - [{org_name}]", exc_info=True)
            raise


    def add_record(self, org_name: str, payload: dict) -> str:
        required = {"name", "url"}
        if not required.issubset(payload.keys()):
            raise ValueError("missing_required_fields_for_insertion")

        record = {
            "name": payload.get("name"),
            "url": payload.get("url")
        }

        if payload.get("fashion_descriptor"):
            collection_name = self._make_collection_name(org_name, True)
            record["fashion_descriptor"] = str(payload["fashion_descriptor"])
        else:
            raise ValueError("missing_fashion_descriptor")

        try:
            collection = self._client.collections.get(collection_name)
            vector = payload.get("vector")
            uid = collection.data.insert(record, vector=vector) if vector else collection.data.insert(record)
            return uid
        except WeaviateConnectionError as e:
            log.critical(f"connection_error_during_insertion - [{org_name}]", exc_info=True)
            raise e
        except Exception as err:
            log.critical(f"insertion_failed - [{org_name}] {collection_name}", exc_info=True)
            return f"insertion_error {err}"

    def delete_record(self, org_name: str, object_uuid: str, image_data: bool = False):
        name = self._make_collection_name(org_name, image_data)
        try:
            collection = self._client.collections.get(name)
            collection.data.delete_by_id(object_uuid)
        except WeaviateConnectionError as e:
            log.critical(f"weaviate_connection_failed - [{org_name}]", exc_info=True)
            raise e
        except Exception:
            log.critical(f"record_deletion_failed - [{org_name}] {name}", exc_info=True)
            raise

    def query_data(self, org_name: str, query_text: str, for_image: bool = False):
        name = self._make_collection_name(org_name, for_image)
        try:
            collection = self._client.collections.get(name)
            result = collection.query.near_text(
                query=query_text,
                limit=MAX_RESULTS,
                return_metadata=MetadataQuery(distance=True),
            )

            if not result:
                return Response(
                    {"message": "No data found in Weaviate"},
                    status=status.HTTP_204_NO_CONTENT,
                )

            return [item.properties for item in result.objects]

        except WeaviateQueryError as e:
            if "no such class" in str(e):
                log.critical(f"no_collection_found - [{org_name}] {name}")
            else:
                log.critical(f"query_failure - [{org_name}]", exc_info=True)
            return []
        except WeaviateConnectionError as e:
            log.critical(f"connection_lost - [{org_name}]", exc_info=True)
            raise e
        except Exception as err:
            log.critical(f"data_retrieval_failed - [{org_name}]", exc_info=True)
            return f"retrieval_error {err}"
