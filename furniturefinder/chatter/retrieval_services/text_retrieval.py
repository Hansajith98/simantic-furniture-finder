import json
from logging import Logger
from tabulate import tabulate
from configparser import ConfigParser

from organization.utils import log_execution_time
from external_services.neo4j_services import Neo4jServices
from external_services.weaviate_services import WeaviateServices, WeaviateServicesForDocument
from chat.models import ChatMessage, RagType

config = ConfigParser()
config.read('conf/application.ini')

NEO4J_URI = config['Neo4j']['uri']
NEO4J_CREDENTIALS = (config['Neo4j']['user'], config['Neo4j']['password'])
RETRIEVAL_INPUT_MAX_ROWS = int(config['Neo4j']['retrieval_input_max_rows'])


class TextRetriever:

    def __init__(self, openAI_client, logger: Logger, neo4j: bool = False) -> None:
        if neo4j:
            self.instance = Neo4jTextRetriever(openAI_client, logger)
        else:
            self.instance = WeaviateTextRetriever(openAI_client, logger)

    @log_execution_time
    def get_product_information(self, retrieval_message: ChatMessage) -> dict:
        return self.instance.get_product_information(retrieval_message)

    def get_token_usage(self) -> dict:
        return self.instance.get_token_usage()


class WeaviateTextRetriever:

    def __init__(self, openAI_client, logger: Logger) -> None:
        self._logger = logger
        self.__prompt_tokens = 0
        self.__completion_tokens = 0
        self._total_tokens = 0
        self.__model = ''
        self.__client = openAI_client

    def get_token_usage(self) -> dict:
        return {
            'prompt_tokens': self.__prompt_tokens,
            'completion_tokens': self.__completion_tokens,
            'total_tokens': self._total_tokens,
            'model': self.__model
        }

    def get_product_information(self, retrieval_message: ChatMessage) -> dict:
        '''Fetches the webpage content for the most relevant URLs
        Params:
            retrieval_message (ChatMessage): Chat message from retrieval assistant
            models (Models): Chat and retrieval models
        Returns:
            Dictionary containing the webpage content
        '''
        empty_search_results_str = \
            'unfortunately, no matching search results were found'
        retrieval_input_max_rows = 15500

        try:
            if retrieval_message.rag_type == RagType.DOCUMENT:
                weaviate_services = WeaviateServicesForDocument()
            else:
                weaviate_services = WeaviateServices()
            search_results = weaviate_services.data_retrieve(
                retrieval_message.organization.name, retrieval_message.message
            )
        except Exception:
            raise

        if not search_results:
            self._logger.info(f'{retrieval_message.request_id} - '
                              f'{retrieval_message.organization.id} - '
                              f'{retrieval_message.session_id} - no_search_results')
            return empty_search_results_str

        search_results = search_results[:retrieval_input_max_rows]

        self._logger.info(f'{retrieval_message.request_id} - '
                          f'{retrieval_message.organization.id} - '
                          f'{retrieval_message.session_id} - '
                          f'product_info_fetched - {len(search_results)}')

        return tabulate(search_results, headers='keys', tablefmt='github')


class Neo4jTextRetriever:
    def __init__(self, openAI_client, logger: Logger) -> None:
        self.__logger = logger
        self.__prompt_tokens = 0
        self.__completion_tokens = 0
        self.__total_tokens = 0
        self.__model = ''
        self.__client = openAI_client

    def get_token_usage(self) -> dict:
        '''
        Return current token usage of instance

        Returns:
        {
            'prompt_tokens': Prompt Tokens,
            'completion_tokens': Completion Tokens,
            'total_tokens': Total Tokens,
            'model': OpenAI model used
        }
        '''
        return {
            'prompt_tokens': self.__prompt_tokens,
            'completion_tokens': self.__completion_tokens,
            'total_tokens': self.__total_tokens,
            'model': self.__model
        }

    def get_embedding(self, query_string: str,
                      model='text-embedding-3-large') -> list:
        '''
        Create Embedding for given query string
        '''

        if not query_string:
            return []

        response = self.__client.embeddings.create(input=query_string,
                                                   model=model)

        self.__prompt_tokens += response.usage.prompt_tokens
        self.__total_tokens += response.usage.total_tokens
        self.__model += response.model

        return response.data[0].embedding

    def is_query_contain_harmful_keywords(self, query: str) -> bool:
        '''
        Check whether given query has harmful keywords like delete, detach, 
        merge, update, insert
        '''
        harmful_keywords = ['delete', 'detach', 'merge', 'update', 'insert']

        return any([(i in query) for i in harmful_keywords])

    def get_product_information(self, retrieval_message: ChatMessage) -> dict:
        '''Fetches the webpage content for the most relevant URLs
        Params:
            retrieval_message (ChatMessage): Chat message from retrieval assistant
            models (Models): Chat and retrieval models
        Returns:
            Dictionary containing the webpage content
        '''
        message = json.loads(retrieval_message.message)
        query = message.get('cypherQuery')
        querySearchString = message.get('querySearchString')
        self.__logger.info(
            f'get_product_information - {querySearchString} - {query}')

        empty_search_results_str = \
            'unfortunately, no matching search results were found'

        if not query:
            self.__logger.info(f'{retrieval_message.request_id} - '
                               f'{retrieval_message.organization.id} - '
                               f'{retrieval_message.session_id} - empty_cypher_query')
            return empty_search_results_str

        elif self.is_query_contain_harmful_keywords(query):
            self.__logger.error(f'{retrieval_message.request_id} - '
                                f'{retrieval_message.organization.id} - '
                                f'{retrieval_message.session_id} - '
                                f'harmful_cypher_query - {query}')
            return empty_search_results_str

        embedding = self.get_embedding(querySearchString)
        try:
            neo4j_services = Neo4jServices(logger=self.__logger.name)

            parameters = {
                'organizationId': str(retrieval_message.organization.id),
                'numberOfNearestNeighbors': RETRIEVAL_INPUT_MAX_ROWS,
                'embedding': embedding
            }

            search_results = neo4j_services.get_query_results(
                query=query, parameters=parameters)

        except Exception:
            raise
        finally:
            if neo4j_services:
                neo4j_services.close()

        if not search_results:
            self.__logger.info(f'{retrieval_message.request_id} - '
                               f'{retrieval_message.organization.id} - '
                               f'{retrieval_message.session_id} - no_search_results')
            return empty_search_results_str

        search_results = search_results[:RETRIEVAL_INPUT_MAX_ROWS]

        self.__logger.info(f'{retrieval_message.request_id} - '
                           f'{retrieval_message.organization.id} - '
                           f'{retrieval_message.session_id} - '
                           f'product_info_fetched - {len(search_results)}')

        return tabulate(search_results, headers='keys', tablefmt='github')
