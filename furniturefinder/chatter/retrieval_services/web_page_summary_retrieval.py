import logging
from configparser import ConfigParser
from typing import Any
import json

config = ConfigParser()
config.read('conf/application.ini')

SUMMARY_MODEL = config['OpenAI']['summary_model']
TEMPERATURE = float(config['OpenAI']['temperature'])
SEED = int(config['OpenAI']['seed'])
DEFAULT_TEXT_EMBEDDING_MODEL = config['OpenAI']['text_embedding_model']


class WebPageSummaryRetrieval():
    def __init__(self, summary_prompt: str, logger_name: str,
                 organization_id: str,
                 embedding_model=DEFAULT_TEXT_EMBEDDING_MODEL) -> None:

        self._prompt = summary_prompt
        self._organization = organization_id
        self._logger = logging.getLogger(logger_name)
        self._embedding_model = embedding_model

    def get_webpage_summary_from_json(self, summary: json) -> str:
        '''
        Extract Web Page Description from Web Page Summary
        '''
        description_words = []

        for key, word in summary.items():
            if word and word not in description_words:
                if isinstance(word, list):
                    word = ', '.join(word)
                item_str = f'{key}: {word}.'
                description_words.append(item_str)

        description = '|'.join(description_words)

        return description

    def get_embedding(self, client: Any, descriptors: dict) -> dict:
        '''
        Extract Embedding for given descriptors

        Returns:
        {
            'vector': Extracted Embedding,
            'prompt_tokens': Prompt Tokens used,
            'completion_tokens': Completion Tokens used,
            'total_tokens': Total Tokens used,
            'model': Model used
        }
        '''
        try:
            response = client.embeddings.create(input=descriptors,
                                                model=self._embedding_model)
        except Exception as error:
            self._logger.error('openai_error - ', exc_info=True)
            raise error

        self._logger.info('embeddings_extracted')

        data = {
            'vector': response.data[0].embedding,
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': 0,
            'total_tokens': response.usage.total_tokens,
            'model': response.model
        }

        return data

    def get_descriptors_from_json_response(self, response) -> dict:
        '''
        Return Web Page Summary from OpenAI json response
        '''
        webpage_summary = {'name': 'Product name from web page',
                           'url': '', 'price': ''}
        json_response_str = response.choices[0].message.content
        self._logger.info(f'json_response_str: {json_response_str}')

        for key, value in json.loads(json_response_str).items():
            webpage_summary[key] = value
        if isinstance(note := webpage_summary.get('additionalNotes'), list):
            webpage_summary['additionalNotes'] = ' '.join(note)
        return webpage_summary

    async def summarizer(self, client: Any, url: str, page_markdown: str) -> dict:
        '''
        Summarize content using a basic prompt loaded from summary_prompt .
        Takes inputs like url, page_title, and tags, returns page_markdown text.
        Returns:
        {
            'message': Created Summary,
            'prompt_tokens': Prompt Tokens used,
            'completion_tokens': Completion Tokens used,
            'total_tokens': Total Tokens used,
            'model': Model used
        }
        '''

        messages = [
            {'role': 'system', 'content': self._prompt},
            {'role': 'user', 'content': f'URL: {url} \n page_markdown: '
             f'{page_markdown}'}
        ]

        try:
            chat_response = client.chat.completions.create(
                model=SUMMARY_MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                seed=SEED,
                max_tokens=2048,
                response_format={'type': 'json_object'}
            )

            web_summary = self.get_descriptors_from_json_response(
                chat_response)

            data = {
                'message': web_summary,
                'prompt_tokens': chat_response.usage.prompt_tokens,
                'completion_tokens': chat_response.usage.completion_tokens,
                'total_tokens': chat_response.usage.total_tokens,
                'model': chat_response.model
            }

            self._logger.info('succeed_web_page_summary_creation')

            return data
        except Exception as e:
            self._logger.error(
                'failed_web_page_summary_creation - ', exc_info=True)
            raise Exception
