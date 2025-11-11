import re
from openai import OpenAI
import logging
from configparser import ConfigParser
from openai.types.chat.chat_completion import ChatCompletion
import os
from dotenv import load_dotenv
import json
import requests
from requests import Response

from django.utils import timezone

from chatter.retrieval_services.image_retrieval import ImageRetriever
from chatter.retrieval_services.text_retrieval import TextRetriever
from chatter.models import ChatMessage, Agent, AgentSubType
from organization.models import Organization, TokenUsage, UsedFunctionName, ProductCollection, \
    Prompt, Configuration, PromptTool, PromptType, PromptTool, ProductWebPage

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

config = ConfigParser()
config.read('conf/application.ini')

RETRIEVAL_MODEL = config['OpenAI']['retrieval_model']
CHAT_MODEL = config['OpenAI']['chat_model']
RETRIEVAL_REPLY_MAX_TOKENS = config['OpenAI']['retrieval_reply_max_tokens']
CHAT_REPLY_MAX_TOKENS = config['OpenAI']['chat_reply_max_tokens']
TEMPERATURE = float(config['OpenAI']['temperature'])
SEED = int(config['OpenAI']['seed'])

client = OpenAI(api_key=OPENAI_API_KEY)

logger = logging.getLogger('semantic_furniture_finder')


def call_api(tool: dict, request_body: dict) -> Response:
    try:
        logger.info(f'calling_api - {tool} - {request_body}')
        url = tool['url']

        match tool['method']:
            case 'GET':
                response = requests.get(
                    url, params=request_body, headers=tool['requestHeader'])
            case 'POST':
                response = requests.post(
                    url, json=request_body, headers=tool['requestHeader'])
            case _:
                raise ValueError(f"Unsupported method: {tool['method']}")

        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        logger.error(
            f'error_calling_api - {url} - {request_body}', exc_info=True)
        raise e
    except Exception:
        logger.error(
            f'error_calling_api - {url} - {request_body}', exc_info=True)
        return Response()


def handle_tool_calls(tool_calls: list, prompt: Prompt) -> list[dict]:
    tools_obj = PromptTool.objects.get(
        prompt=prompt, prompt_type=PromptType.CHAT_PROMPT, is_delete=False).tools
    available_tools = {}
    for tool in tools_obj:
        function = tool['openai_tool_data']['function']
        available_tools[function['name']] = {
            'url': tool['apiUrl'],
            'requestHeader': tool.get('requestHeader', {}),
            'method': tool.get('method', 'GET')
        }

def get_product_urls_from_message(message: str) -> list[str]:
    '''
    Get all product URLs in the assistant message
    '''
    URL_PATTERN = r'(https?://(?:www\.)?[\w-]+(?:\.[\w-]+)+(?:/[\w\-./?%&=]*)?)'

    product_urls = re.findall(URL_PATTERN, message)
    return product_urls


def validate_product_urls(organization: Organization, product_urls: list[str]) -> bool:
    '''
    Validate product URLs in the assistant message
    '''
    for url in product_urls:
        product = ProductWebPage.objects.filter(
            url=url, organization=organization)
        if not product.exists():
            logger.warning(f'product_url_not_in_db - {url}')
            return False
    return True


def get_response(user_message: ChatMessage, messages: list[ChatMessage],
                 prompts: Prompt, enable_neo4j: bool) -> ChatMessage:
    '''
    Get response for the user message
    '''

    retrieved_information = retrieve_information(user_message,
                                                 messages, prompts, enable_neo4j)

    chat_message = get_chat_response(user_message, retrieved_information,
                                     messages['chat'], prompts)

    while (tool_calls := chat_message.message.get('tool_calls')):
        logger.info(f'calling_tools - {tool_calls}')
        messages['chat'].append(chat_message.message)

        logger.info(f'[{user_message.request_id}] - [{user_message.organization.id}] - '
                    f'calling_tools - {tool_calls}')
        tool_results = handle_tool_calls(tool_calls, prompts.id)
        tool_result_msg = ChatMessage(
            request_id=chat_message.request_id,
            user_id=chat_message.user_id,
            organization_id=chat_message.organization_id,
            session_id=chat_message.session_id,
            timestamp=timezone.now(),
            agent=Agent.ASSISTANT,
            agent_subtype=AgentSubType.TOOL_ASSISTANT,
            message=tool_results,
            is_image=False,
            image_url=None)

        tool_result_msg.save()
        messages['chat'].append(tool_result_msg.message)

        chat_message = get_chat_response(user_message, retrieved_information,
                                         messages['chat'], prompts)
        logger.info(f'chat_response_got_from_openai - {chat_message.message}')

    return chat_message


def get_collections_as_str(organization_id: str) -> str:
    '''
    Fetch and return available collections except Default Collection

    Returns:
    Collection Name 1: URL 1 \\n
    Collection Name 2: URL 2 \\n
    '''
    collections_obj = ProductCollection.objects.filter(
        organization=organization_id, is_delete=False)
    collections_list = []
    for collection in collections_obj:
        if collection.name == 'Default Collection':
            continue
        collections_list.append(f'{collection.name}: {collection.url}')

    if collections_list:
        return '\\n'.join(collections_list)
    else:
        return 'there are no collections available.'


def retrieve_information(user_message: ChatMessage, messages: list[dict],
                         prompts: Prompt, neo4j: bool) -> dict:
    '''
    Get information required to process user message
    '''

    if user_message.is_image and Configuration.objects.get(
            organization=user_message.organization.id,
            is_delete=False).image_summary_required:
        image_retriever = ImageRetriever(
            prompt=prompts.image_desc_prompt, neo4j=neo4j)
        retrieved_information = \
            image_retriever.match_images(client, user_message)

        token_usage = image_retriever.get_token_usage()

        logger.info(f'[{user_message.request_id}] - [{user_message.organization.id}] - '
                    f'[{user_message.session_id}] - image_chat_response_created')

    else:
        if neo4j:
            collections = get_collections_as_str(user_message.organization.id)
            prompt = prompts.graph_retrieval_prompt.replace('<collections>',
                                                            collections)
        else:
            prompt = prompts.vector_rephrase_prompt

        retrieval_message = get_retrieval_response(
            user_message, messages['retrieval'], prompt)

        text_retriever = TextRetriever(
            logger=logger, neo4j=neo4j, openAI_client=client)
        retrieved_information = text_retriever.get_product_information(
            retrieval_message)

        token_usage = text_retriever.get_token_usage()

        logger.info(f'[{user_message.request_id}] - [{user_message.organization.id}] - '
                    f'[{user_message.session_id}] - text_chat_response_created')

    token_usage['request_id'] = user_message.request_id
    token_usage['used_function_name'] = UsedFunctionName.USER_MESSAGE_REPHRASER
    update_token_usage(user_message.organization.id, token_usage)

    return retrieved_information


def update_token_usage(organization_id: str, token_usage: dict) -> None:
    '''
    Update token usage on db
    '''
    TokenUsage.objects.create(
        organization_id=organization_id,
        input_tokens=token_usage.get('input_tokens', 0),
        completion_tokens=token_usage.get('completion_tokens', 0),
        total_tokens=token_usage.get('total_tokens', 0),
        model=token_usage.get('model', ''),
        used_function_name=token_usage.get('used_function_name'),
        request_id=token_usage.get('request_id')
    )


def get_retrieval_response(
        user_message: ChatMessage,
        messages: list,  prompt: str) -> ChatMessage:
    '''
    Gets the most relevant URLs from the retrieval assistant
    '''

    retrieval_response = get_openai_response(
        user_message, messages, prompt, 'retrieval')

    update_token_usage(user_message.organization.id, {
        'prompt_tokens': retrieval_response.usage.prompt_tokens,
        'completion_tokens': retrieval_response.usage.completion_tokens,
        'total_tokens': retrieval_response.usage.total_tokens,
        'model': retrieval_response.model,
        'used_function_name': UsedFunctionName.USER_MESSAGE_REPHRASER,
        'request_id': user_message.request_id
    })

    logger.info(f'[{user_message.request_id}] - [{user_message.organization.id}] - '
                f'[{user_message.session_id}] - got_retrieval_response - {retrieval_response.choices[0].message.content}')

    return get_retrieval_message(user_message, retrieval_response)


def get_openai_response(user_message: ChatMessage, messages: list,
                        system_prompt: str, request_type: str) -> ChatCompletion:
    '''
    Gets the response from OpenAI
    '''

    try:
        messages = [{'role': 'system', 'content': system_prompt}] + messages
        tools = None

        if request_type == 'retrieval':
            model = RETRIEVAL_MODEL
            max_reply_tokens = RETRIEVAL_REPLY_MAX_TOKENS
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=TEMPERATURE,
                max_tokens=int(max_reply_tokens), seed=int(SEED),
                response_format={'type': 'json_object'})
        elif request_type == 'chat':
            model = CHAT_MODEL
            max_reply_tokens = CHAT_REPLY_MAX_TOKENS
            if tools_obj := PromptTool.objects.filter(
                    prompt__organization=user_message.organization,
                    prompt_type=PromptType.CHAT_PROMPT, is_delete=False).first():
                tools = []
                for tool in tools_obj.tools:
                    tools.append(tool['openai_tool_data'])

            response = client.chat.completions.create(
                model=model, messages=messages, temperature=TEMPERATURE,
                max_tokens=int(max_reply_tokens), seed=int(SEED),
                response_format={'type': 'json_object'},
                tools=tools)
        logger.info(f'[{user_message.request_id}] - [{user_message.organization.id}] - '
                    f'[{user_message.session_id}] - openai_request_sent {"- tools_enabled" if tools else ""}')

    except Exception as error:
        logger.error(f'[{user_message.request_id}] - [{user_message.organization.id}] - '
                     f'[{user_message.session_id}] - openai_error - ',
                     exc_info=True)
        raise error

    logger.info(f'[{user_message.request_id}] - [{user_message.organization.id}] - '
                f'[{user_message.session_id}] - openai_response_received - '
                f'{request_type}')

    return response


def get_retrieval_message(user_message: ChatMessage,
                          retrieval_response: any) -> ChatMessage:
    '''
    Creates retrieval chat message for assistant's reply based on response 
    from OpenAI
    '''

    retrieval_message = ChatMessage(
        request_id=user_message.request_id,
        user_id=user_message.user_id,
        organization_id=user_message.organization_id,
        session_id=user_message.session_id,
        timestamp=timezone.now(),
        agent=Agent.ASSISTANT,
        agent_subtype=AgentSubType.RETRIEVAL_ASSISTANT,
        rag_type=user_message.rag_type,
        model=retrieval_response.model,
        prompt_tokens=retrieval_response.usage.prompt_tokens,
        completion_tokens=retrieval_response.usage.completion_tokens,
        message=retrieval_response.choices[0].message.content,
        is_image=False,
        image_url=None)

    retrieval_message.save()
    return retrieval_message


def get_chat_response(user_message: ChatMessage, graph_information: dict,
                      messages: list[ChatMessage], prompts: Prompt) -> ChatMessage:
    '''
    Creates the chat prompt and gets the chat assistant's response for a 
    user message
    '''

    chat_system_prompt = get_chat_system_prompt(
        user_message, graph_information, prompts)
    chat_response = get_openai_response(user_message, messages,
                                        chat_system_prompt, 'chat')

    logger.info(f'[{user_message.request_id}] - [{user_message.organization.id}] - '
                f'[{user_message.session_id}] - got_chat_response - '
                f'{chat_response.choices[0]}')

    update_token_usage(user_message.organization.id, {
        'prompt_tokens': chat_response.usage.prompt_tokens,
        'completion_tokens': chat_response.usage.completion_tokens,
        'total_tokens': chat_response.usage.total_tokens,
        'model': chat_response.model,
        'used_function_name': UsedFunctionName.ASSISTANT_MESSAGE_GENERATOR,
        'request_id': user_message.request_id
    })
    logger.info(f'[{user_message.request_id}] - [{user_message.organization.id}] - '
                f'[{user_message.session_id}] - created_chat_response')

    return get_chat_message(user_message, chat_response)


def get_chat_system_prompt(user_message: ChatMessage,
                           retrieved_information: dict, prompts: Prompt) -> str:
    '''
    Gets the chat assistant system prompt
    '''

    chat_system_prompt = prompts.chat_response_prompt

    if not retrieved_information:
        logger.info(f'[{user_message.request_id}] - [{user_message.organization_id}] - '
                    f'[{user_message.session_id}] - '
                    'no_additional_information_added_to_prompt')
        return chat_system_prompt

    chat_system_prompt = chat_system_prompt.replace(
        '<product-information>', retrieved_information)

    chat_system_prompt = chat_system_prompt.replace(
        '<collections>', get_collections_as_str(user_message.organization.id))

    chat_system_prompt = chat_system_prompt.replace(
        '<current-date>', timezone.now().strftime('%Y-%m-%d'))

    logger.info(f'[{user_message.request_id}] - [{user_message.organization_id}] - '
                f'[{user_message.session_id}] - '
                'retrieved_information_added_to_prompt')

    return chat_system_prompt


def format_assistant_reply_content(response: any) -> dict:
    '''
    Format openAI reply with role, content, tool_calls parameters as openAI
    standard message
    '''
    if response.choices[0].message.tool_calls:
        return {
            'role': response.choices[0].message.role,
            'tool_calls': [
                {
                    'id': i.id,
                    'type': i.type,
                    'function': {
                        'name': i.function.name,
                        'arguments': i.function.arguments,
                    }
                } for i in response.choices[0].message.tool_calls
            ]
        }
    else:
        return {
            'role': response.choices[0].message.role,
            'content': [
                {
                    'text': response.choices[0].message.content,
                    'type': 'json_object'
                }
            ]
        }


def get_chat_message(user_message: ChatMessage,
                     chat_response: any) -> ChatMessage:
    '''
    Creates chat message for assistant's reply based on response from OpenAI
    '''
    message = format_assistant_reply_content(chat_response)

    chat_message = ChatMessage(
        request_id=user_message.request_id,
        user_id=user_message.user_id,
        organization_id=user_message.organization_id,
        session_id=user_message.session_id,
        timestamp=timezone.now(),
        agent=Agent.ASSISTANT,
        agent_subtype=AgentSubType.CHAT_ASSISTANT,
        model=chat_response.model,
        prompt_tokens=chat_response.usage.prompt_tokens,
        completion_tokens=chat_response.usage.completion_tokens,
        message=message,
        rag_type=user_message.rag_type,
        is_image=False,
        image_url=None)

    chat_message.save()
    return chat_message


def get_embedding(log_str: str, text: str) -> dict:
    '''
    Extract Embedding for given descriptors
    '''

    try:
        response = client.embeddings.create(
            input=text, model='text-embedding-3-large')
    except Exception as error:
        logger.error(f'{log_str} - openai_error |', exc_info=True)
        raise error

    data = {
        'vector': response.data[0].embedding,
        'prompt_tokens': response.usage.prompt_tokens,
        'completion_tokens': 0,
        'total_tokens': response.usage.total_tokens,
        'model': response.model
    }

    return data
