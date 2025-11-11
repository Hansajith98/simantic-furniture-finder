import asyncio
import string
import json
import logging
from logging import Logger
import base64
from datetime import datetime
from random import choices
import requests
from tabulate import tabulate
from PIL import Image
from configparser import ConfigParser

from chatter.models import ChatMessage, Agent
from organization.utils import log_execution_time
from external_services.weaviate_services import WeaviateServices

config = ConfigParser()
config.read('conf/application.ini')

IMAGE_DESCRIPTOR_MODEL = config['OpenAI']['image_descriptor_model']
MAX_TOKENS = int(config['OpenAI']['image_description_max_tokens'])
TEMPERATURE = float(config['OpenAI']['image_description_temperature'])
SEED = int(config['OpenAI']['image_description_seed'])

WEAVIATE_RESULT_LIMIT = config['Weaviate']['result_limit']
RESIZING_RESOLUTION = (512, 768)

DEFAULT_TEXT_EMBEDDING_MODEL = config['OpenAI']['text_embedding_model']


class ImageRetriever:

    def __init__(self, prompt, resizing_resolution=RESIZING_RESOLUTION,
                 image_directory='data', descriptor_model=IMAGE_DESCRIPTOR_MODEL,
                 logger=None, embedding_model=DEFAULT_TEXT_EMBEDDING_MODEL):

        self.__prompt = prompt
        self.__resizing_resolution = resizing_resolution

        self.__image_directory = image_directory
        self.__descriptor_model = descriptor_model
        self.__embedding_model = embedding_model

        self.__max_shortlist_count = 7

        self.__prompt_tokens = 0
        self.__completion_tokens = 0
        self.__total_tokens = 0
        self.__model = ''
        if not isinstance(logger, Logger):
            self.__logger = logging.getLogger('chat_with_website')
        else:
            self.__logger = logger

    def get_token_usage(self) -> dict:
        '''
        Return current Token Usage of Image Retriever
        '''
        return {
            'prompt_tokens': self.__prompt_tokens,
            'completion_tokens': self.__completion_tokens,
            'total_tokens': self.__total_tokens,
            'model': self.__model
        }

    def download_image(self, message: ChatMessage) -> str:
        '''
        Downloads an image from specified URL and saves it to data folder
        '''

        timestamp_str = datetime.strftime(
            message.timestamp, '%Y%m%d_%H_%M_%S_%f')
        random_str = ''.join(choices(string.ascii_lowercase, k=5))

        response = requests.get(message.image_url)
        filename = f'{timestamp_str}_{random_str}.png'

        with open(f'{self.__image_directory}/{filename}', 'wb') as file:
            file.write(response.content)

            self.__logger.info(f'[{message.request_id}] - [{message.organization_id}] | '
                               f'[{message.session_id}] | image_downloaded | {filename}')

        return filename

    def resize_image(self, log_str: str, filename: str,
                     resizing_resolution: tuple = None) -> str:
        '''
        Resize image 
        '''
        if not resizing_resolution:
            resizing_resolution = self.__resizing_resolution

        with Image.open(f'{self.__image_directory}/{filename}') as image:
            image.thumbnail(resizing_resolution)
            resized_image_filename = f'{self.__image_directory}/r{filename}'
            image.save(resized_image_filename)

            self.__logger.info(f'{log_str} | image_resized | '
                               f'{resized_image_filename}')

        return resized_image_filename

    def get_descriptors_from_json_response(self, response) -> str:
        '''
        Return Image Descriptor from OpenAI json response
        '''
        image_descriptors = {'name': 'Target Fashion Item in Image',
                             'url': '', 'price': ''}
        json_response_str = response.choices[0].message.content

        for key, value in json.loads(json_response_str).items():
            image_descriptors[key] = value

        return image_descriptors

    def get_image_descriptors(self, log_str: str, client: any,
                              image_filename: str) -> dict:
        '''
        Get Image Descriptor from OpenAI
        '''
        with open(image_filename, 'rb') as file:
            encoded_content = base64.b64encode(file.read()).decode('utf-8')
            image_url = f'data:image/jpeg;base64,{encoded_content}'

        try:
            response = client.chat.completions.create(
                model=self.__descriptor_model,
                messages=[
                    {'role': 'system', 'content': [
                        {'type': 'text', 'text': self.__prompt}]},
                    {'role': 'user', 'content': [
                        {'type': 'image_url', 'image_url': {'url': image_url}}]}
                ], response_format={'type': 'json_object'},
                temperature=TEMPERATURE, max_tokens=MAX_TOKENS, seed=SEED
            )
            self.__prompt_tokens += response.usage.prompt_tokens
            self.__completion_tokens += response.usage.completion_tokens
            self.__total_tokens += response.usage.total_tokens
            self.__model += response.model

        except Exception as error:
            self.__logger.error(f'{log_str} | openai_error |', exc_info=True)
            raise error

        self.__logger.info(
            f'{log_str} | image_description_received_from_openai')

        image_descriptors = self.get_descriptors_from_json_response(response)

        return image_descriptors

    def get_embedding(self, log_str: str, client: any, descriptors: dict) -> dict:
        '''
        Extract Embedding for given descriptors
        '''
        embedding_words = []

        for word in descriptors.values():
            if word and word not in embedding_words:
                embedding_words.append(word)

        embedding_str = '|'.join(embedding_words)

        try:
            response = client.embeddings.create(
                input=embedding_str, model=self.__embedding_model)
        except Exception as error:
            self.__logger.error(f'{log_str} | openai_error |', exc_info=True)
            raise error

        self.__logger.info(f'{log_str} | embeddings_extracted')

        self.__prompt_tokens += response.usage.prompt_tokens
        self.__total_tokens += response.usage.total_tokens
        self.__model += response.model

        data = {
            'vector': response.data[0].embedding,
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': 0,
            'total_tokens': response.usage.total_tokens,
            'model': response.model
        }

        return data

    def get_shortlisted_items_table(self, result: list,
                                    target_descriptors: str) -> str:
        '''
        Get short list items and return table items 
        '''
        shortlisted_dresses = []
        urls = set()

        for row in result:
            url = row['url']
            name = row['name']

            if url in urls:
                continue

            urls.add(url)
            shortlisted_dress = {'name': name, 'url': url}

            fashion_description = i if (i := row.get(
                'furnitureDescriptors')) else row.get('fashion_descriptor')

            for key, value in json.loads(fashion_description).items():
                shortlisted_dress[key] = value

            shortlisted_dresses.append(shortlisted_dress)

            if len(shortlisted_dresses) >= self.__max_shortlist_count:
                break

        return [target_descriptors] + shortlisted_dresses

    def get_weaviate_query_results(self, message: ChatMessage,
                                   image_descriptors: dict) -> list:
        '''
        Retrieve matching data from Weaviate
        '''
        try:
            weaviate_services = WeaviateServices()
            results = weaviate_services.data_retrieve(
                message.organization.name, json.dumps(image_descriptors), True
            )
            shortlisted_items = self.get_shortlisted_items_table(
                results, image_descriptors)
            return shortlisted_items

        except Exception:
            self.__logger.error(
                'weaviate data retrieve failed error', exc_info=True)
            raise

    @log_execution_time
    def match_images(self, client, message: ChatMessage) -> str:
        '''
        Match images and retrieve shortlisted items.
        '''
        log_str = (f'[{message.request_id}] | [{message.organization.id}] | '
                   f'[{message.session_id}]')

        image_descriptors = asyncio.run(
            self.preprocess_n_get_image_description(
                log_str, client, message, is_chat=True))

        embedding = self.get_embedding(
            log_str, client, image_descriptors).get('vector')

        shortlisted_results = self.get_weaviate_query_results(
                message, image_descriptors)

        shortlisted_table = tabulate(
            shortlisted_results, headers='keys', tablefmt='github')

        self.__logger.info(f'[{message.request_id}] - [{message.organization_id}] | '
                           f'[{message.session_id}] | shortlisted_results_fetched')

        return shortlisted_table

    async def preprocess_n_get_image_description(self, log_str: str, client,
                                                 message: ChatMessage,
                                                 is_chat: bool = False) -> dict:
        '''
        Preprocess and retrieve image description.
        '''
        filename = self.download_image(message)
        if is_chat:
            resized_image_filename = self.resize_image(log_str, filename)
        else:
            resized_image_filename = self.resize_image(log_str, filename,
                                                       (365, 365))

        image_descriptors = self.get_image_descriptors(
            log_str, client, resized_image_filename)

        return image_descriptors
