import re
import os
import json
import logging
import requests
from requests import Response
from configparser import ConfigParser
from dotenv import load_dotenv
from datetime import datetime
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion

from django.utils import timezone

from dialogue.retrieval_services.visual_retriever import ImageRetrieverService
from dialogue.retrieval_services.text_retriever import TextRetrieverService
from dialogue.models import ChatEntry, Role, RoleSubtype
from organization.models import (
    OrganizationEntity, TokenLog, ExecutedFunction, ProductCategory,
    PromptTemplate, ConfigurationSetting, PromptIntegration, PromptType,
    ProductPage
)

load_dotenv()
OPENAI_KEY = os.getenv('OPENAI_API_KEY')

config = ConfigParser()
config.read('conf/application.ini')

RETRIEVAL_MODEL_NAME = config['OpenAI']['retrieval_model']
CHAT_MODEL_NAME = config['OpenAI']['chat_model']
RETRIEVAL_REPLY_LIMIT = config['OpenAI']['retrieval_reply_max_tokens']
CHAT_REPLY_LIMIT = config['OpenAI']['chat_reply_max_tokens']
MODEL_TEMPERATURE = float(config['OpenAI']['temperature'])
RANDOM_SEED = int(config['OpenAI']['seed'])

openai_client = OpenAI(api_key=OPENAI_KEY)

app_logger = logging.getLogger('semantic_furniture_finder')


def perform_external_call(api_details: dict, payload: dict) -> Response:
    try:
        app_logger.info(f'api_request_initiated - {api_details} - {payload}')
        api_url = api_details['url']

        match api_details['method']:
            case 'GET':
                result = requests.get(api_url, params=payload, headers=api_details['requestHeader'])
            case 'POST':
                result = requests.post(api_url, json=payload, headers=api_details['requestHeader'])
            case _:
                raise ValueError(f"Unsupported HTTP method: {api_details['method']}")

        result.raise_for_status()
        return result

    except requests.exceptions.RequestException:
        app_logger.error(f'api_call_failed - {api_url} - {payload}', exc_info=True)
        raise
    except Exception:
        app_logger.error(f'api_call_exception - {api_url} - {payload}', exc_info=True)
        return Response()


def extract_product_links(message_text: str) -> list[str]:
    pattern = r'(https?://(?:www\.)?[\w-]+(?:\.[\w-]+)+(?:/[\w\-./?%&=]*)?)'
    return re.findall(pattern, message_text)


def verify_product_links(org_entity: OrganizationEntity, urls: list[str]) -> bool:
    for link in urls:
        if not ProductPage.objects.filter(url=link, organization=org_entity).exists():
            app_logger.warning(f'unknown_product_url - {link}')
            return False
    return True


def generate_response(user_entry: ChatEntry, chat_context: list[ChatEntry],
                      prompts: PromptTemplate, enable_graph_db: bool) -> ChatEntry:
    context_info = fetch_context_data(user_entry, chat_context, prompts, enable_graph_db)

    assistant_entry = generate_chat_output(user_entry, context_info, chat_context['chat'], prompts)
    return assistant_entry


def list_collections_as_string(org_id: str) -> str:
    collections = ProductCategory.objects.filter(organization=org_id, is_delete=False)
    formatted = [f'{col.name}: {col.url}' for col in collections if col.name != 'Default Collection']
    return '\n'.join(formatted) if formatted else 'no collections available.'


def fetch_context_data(user_entry: ChatEntry, chat_context: list[dict],
                       prompts: PromptTemplate, enable_graph: bool) -> dict:
    if user_entry.is_image and ConfigurationSetting.objects.get(
            organization=user_entry.organization.id, is_delete=False).image_summary_required:
        image_retriever = ImageRetrieverService(prompts.image_desc_prompt, enable_graph)
        context_data = image_retriever.match_images(openai_client, user_entry)
        token_usage = image_retriever.get_token_summary()

    else:
        if enable_graph:
            collections_info = list_collections_as_string(user_entry.organization.id)
            updated_prompt = prompts.graph_retrieval_prompt.replace('<collections>', collections_info)
        else:
            updated_prompt = prompts.vector_rephrase_prompt

        retrieval_entry = perform_retrieval_query(user_entry, chat_context['retrieval'], updated_prompt)
        text_retriever = TextRetrieverService(openAI_client=openai_client, logger=app_logger, neo4j=enable_graph)
        context_data = text_retriever.get_product_details(retrieval_entry)
        token_usage = text_retriever.get_token_summary()

    token_usage['request_id'] = user_entry.request_id
    token_usage['used_function_name'] = ExecutedFunction.USER_MESSAGE_REPHRASER
    record_token_usage(user_entry.organization.id, token_usage)

    return context_data


def record_token_usage(org_id: str, token_log: dict) -> None:
    TokenLog.objects.create(
        organization_id=org_id,
        input_tokens=token_log.get('prompt_tokens', 0),
        completion_tokens=token_log.get('completion_tokens', 0),
        total_tokens=token_log.get('total_tokens', 0),
        model=token_log.get('model', ''),
        used_function_name=token_log.get('used_function_name'),
        request_id=token_log.get('request_id')
    )


def perform_retrieval_query(user_entry: ChatEntry, chat_history: list, prompt_text: str) -> ChatEntry:
    response = query_openai_api(user_entry, chat_history, prompt_text, 'retrieval')
    record_token_usage(user_entry.organization.id, {
        'prompt_tokens': response.usage.prompt_tokens,
        'completion_tokens': response.usage.completion_tokens,
        'total_tokens': response.usage.total_tokens,
        'model': response.model,
        'used_function_name': ExecutedFunction.USER_MESSAGE_REPHRASER,
        'request_id': user_entry.request_id
    })

    return construct_retrieval_entry(user_entry, response)


def query_openai_api(user_entry: ChatEntry, conversation: list,
                     sys_prompt: str, query_type: str) -> ChatCompletion:
    try:
        messages = [{'role': 'system', 'content': sys_prompt}] + conversation
        tools = None

        if query_type == 'retrieval':
            response = openai_client.chat.completions.create(
                model=RETRIEVAL_MODEL_NAME, messages=messages,
                temperature=MODEL_TEMPERATURE,
                max_tokens=int(RETRIEVAL_REPLY_LIMIT), seed=RANDOM_SEED,
                response_format={'type': 'json_object'}
            )
        else:
            if integration := PromptIntegration.objects.filter(
                    prompt__organization=user_entry.organization,
                    prompt_type=PromptType.CHAT_PROMPT, is_delete=False).first():
                tools = [tool['openai_tool_data'] for tool in integration.tools]

            response = openai_client.chat.completions.create(
                model=CHAT_MODEL_NAME, messages=messages,
                temperature=MODEL_TEMPERATURE,
                max_tokens=int(CHAT_REPLY_LIMIT), seed=RANDOM_SEED,
                response_format={'type': 'json_object'},
                tools=tools
            )

        app_logger.info(f'[{user_entry.request_id}] - [{user_entry.session_id}] - '
                        f'openai_query_sent {"(with_tools)" if tools else ""}')
        return response

    except Exception:
        app_logger.error(f'[{user_entry.request_id}] - openai_query_failed', exc_info=True)
        raise


def construct_retrieval_entry(user_entry: ChatEntry, api_result: any) -> ChatEntry:
    entry = ChatEntry(
        request_id=user_entry.request_id,
        user_id=user_entry.user_id,
        organization_id=user_entry.organization_id,
        session_id=user_entry.session_id,
        timestamp=timezone.now(),
        role=Role.ASSISTANT,
        role_subtype=RoleSubtype.RETRIEVAL_ASSISTANT,
        rag_type=user_entry.rag_type,
        model=api_result.model,
        prompt_tokens=api_result.usage.prompt_tokens,
        completion_tokens=api_result.usage.completion_tokens,
        message=api_result.choices[0].message.content,
        is_image=False
    )
    entry.save()
    return entry


def generate_chat_output(user_entry: ChatEntry, context_info: dict,
                         chat_log: list, prompt_template: PromptTemplate) -> ChatEntry:
    system_prompt = compose_chat_prompt(user_entry, context_info, prompt_template)
    chat_response = query_openai_api(user_entry, chat_log, system_prompt, 'chat')

    record_token_usage(user_entry.organization.id, {
        'prompt_tokens': chat_response.usage.prompt_tokens,
        'completion_tokens': chat_response.usage.completion_tokens,
        'total_tokens': chat_response.usage.total_tokens,
        'model': chat_response.model,
        'used_function_name': ExecutedFunction.ASSISTANT_MESSAGE_GENERATOR,
        'request_id': user_entry.request_id
    })

    return compose_chat_entry(user_entry, chat_response)


def compose_chat_prompt(user_entry: ChatEntry, context_info: dict,
                        prompt_template: PromptTemplate) -> str:
    prompt_text = prompt_template.chat_response_prompt

    if not context_info:
        return prompt_text

    prompt_text = prompt_text.replace('<product-information>', context_info)
    prompt_text = prompt_text.replace('<collections>', list_collections_as_string(user_entry.organization.id))
    prompt_text = prompt_text.replace('<current-date>', timezone.now().strftime('%Y-%m-%d'))

    return prompt_text


def format_openai_message(response_obj: any) -> dict:
    msg = response_obj.choices[0].message
    if msg.tool_calls:
        return {
            'role': msg.role,
            'tool_calls': [
                {
                    'id': t.id,
                    'type': t.type,
                    'function': {
                        'name': t.function.name,
                        'arguments': t.function.arguments
                    }
                } for t in msg.tool_calls
            ]
        }
    else:
        return {'role': msg.role, 'content': [{'text': msg.content, 'type': 'json_object'}]}


def compose_chat_entry(user_entry: ChatEntry, chat_output: any) -> ChatEntry:
    structured_msg = format_openai_message(chat_output)
    entry = ChatEntry(
        request_id=user_entry.request_id,
        user_id=user_entry.user_id,
        organization_id=user_entry.organization_id,
        session_id=user_entry.session_id,
        timestamp=timezone.now(),
        role=Role.ASSISTANT,
        role_subtype=RoleSubtype.CHAT_ASSISTANT,
        model=chat_output.model,
        prompt_tokens=chat_output.usage.prompt_tokens,
        completion_tokens=chat_output.usage.completion_tokens,
        message=structured_msg,
        rag_type=user_entry.rag_type,
        is_image=False
    )
    entry.save()
    return entry


def create_text_embedding(log_prefix: str, content: str) -> dict:
    try:
        response = openai_client.embeddings.create(
            input=content, model='text-embedding-3-large'
        )
    except Exception:
        app_logger.error(f'{log_prefix} - embedding_error', exc_info=True)
        raise

    return {
        'vector': response.data[0].embedding,
        'prompt_tokens': response.usage.prompt_tokens,
        'completion_tokens': 0,
        'total_tokens': response.usage.total_tokens,
        'model': response.model
    }
