import json
import logging
import re
from datetime import timedelta
from configparser import ConfigParser
import tiktoken

from django.db.models.query import QuerySet
from rest_framework.views import APIView
from django.db import transaction
from rest_framework.response import Response
from rest_framework.request import Request
from rest_framework import status
from django.utils import timezone

from chatter.utils import get_response
from chatter.models import ChatMessage, Agent, AgentSubType, RagType
from chatter.serializer import ChatHistorySerializer
from organization.models import Organization, Prompt
from organization.models import TokenUsage
from organization.utils import log_execution_time


logger = logging.getLogger('semantic_furniture_finder')

URL_PATTERN = r'(https?://(?:www\.)?[\w-]+(?:\.[\w-]+)+(?:/[\w\-./?%&=]*)?)'

config = ConfigParser()
config.read('conf/application.ini')

RETRIEVAL_HISTORY_LIMIT = int(
    config['Conversation']['retrieval_history_limit'])
CHAT_HISTORY_LIMIT = int(config['Conversation']['chat_history_limit'])
RETRIEVAL_HISTORY_DURATION = int(config['Conversation']['history_duration'])

MAX_HISTORY_LIMIT = max(CHAT_HISTORY_LIMIT, RETRIEVAL_HISTORY_LIMIT)
MESSAGE_INPUT_MAX_TOKENS = int(config['OpenAI']['message_input_max_tokens'])


class TokenCountExceededException(Exception):
    '''Exception for raise when Token Count exceed'''

    def __init__(self):
        super().__init__('Token count exceeded the limit')


class ChatterView(APIView):

    @transaction.atomic
    def post(self, request: Request) -> Response:
        try:
            request_data = request.data
            logger.info(f'chat_request_received - [{request_data}]')

            if not request_data:
                logger.info(f"no_request_data : [{request_data}]")
                return Response({"message": "No data provided in the request."},
                                status=status.HTTP_400_BAD_REQUEST)

            required_fields = ["message",
                               "userId", "sessionId", "requestId"]

            if not all(item in request_data.keys() for item in required_fields):
                logger.info(f"no_required_data_fields : [{request_data}]")
                return Response({"message": "Missing required field"},
                                status=status.HTTP_400_BAD_REQUEST)

            user_message = get_user_message(request_data)
            user_message.save()

            messages = get_message_history(user_message)

            assistant_message = get_response(
                user_message, messages)
            assistant_message.save()
            logger.info(f'[{user_message.request_id}] - [{user_message.session_id}] - chat_history_saved')

            token_info_obj = TokenUsage.objects.filter(
                request_id=request_data.get("requestId"), is_delete=False)

            token_info = []

            for row in token_info_obj:
                token_info.append(
                    {
                        'inputToken': row.input_tokens,
                        'outputToken': row.completion_tokens,
                        'totalToken': row.total_tokens,
                        'model': row.model,
                        'function': row.used_function_name
                    }
                )
            message = get_chat_reply_from_message(assistant_message.message)

            return Response({
                "userId": request_data["userId"],
                "sessionId": request_data["sessionId"],
                "content": {
                    "message": message,
                    "suggestions": "",
                    "reference": "",
                    "tokenInfo": token_info,
                }
            },
                status=status.HTTP_201_CREATED)
        except TokenCountExceededException:
            transaction.set_rollback(True)
            logger.error("internal_server_error", exc_info=True)
            return Response({"message": "Message Token Count Exceeded."},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except:
            transaction.set_rollback(True)
            logger.error("internal_server_error", exc_info=True)
            return Response({"message": "Internal Server Error"},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    
def get_chat_reply_from_message(message: str) -> str:
    message_content = json.loads(message.get('content')[0].get('text'))
    chat_reply = message_content.get('chatReply')
    return chat_reply


@log_execution_time
def get_message_history(user_message: ChatMessage) -> dict:
    '''
    Fetches the retrieval and chat message histories from the database

    Returns:
        Dictionary containing the chat and retrieval assistant message histories
    '''

    previous_timestamp, timestamp = get_timestamp_intervals(
        user_message)
    rows = ChatMessage.objects.filter(
        user_id=user_message.user_id,
        session_id=user_message.session_id,
        rag_type=user_message.rag_type,
        timestamp__range=(previous_timestamp, timestamp)
    ).order_by('-timestamp')

    retrieval_history, chat_history = get_separate_message_histories(rows)
    logger.info(f'retrieval - {retrieval_history}')
    logger.info(f'chat - {chat_history}')

    chat_history = get_modified_chat_history_for_image_input(
        user_message, chat_history)

    chat_history = trim_message_history_with_safe_token_limit(chat_history)
    if not chat_history:
        raise TokenCountExceededException

    retrieval_history = retrieval_history[-len(chat_history):]

    logger.debug(f'conversation_history_fetched - r:{retrieval_history} - '
                 f'ch:{chat_history}')

    logger.info(f'conversation_history_fetched - r:{len(retrieval_history)} - '
                f'ch:{len(chat_history)}')

    return {'retrieval': retrieval_history, 'chat': chat_history}


def trim_message_history_with_safe_token_limit(
        messages: list, max_tokens: int = MESSAGE_INPUT_MAX_TOKENS) -> list:
    '''
    Trim message history to stay within a safe token limit.
    '''
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    token_count = len(encoding.encode(
        ''.join(item['content'] for item in messages)))
    trimmed_messages = messages

    while token_count > max_tokens and len(trimmed_messages) > 0:
        trimmed_messages = trimmed_messages[1:]
        token_count = len(encoding.encode(
            ''.join(item['content'] for item in trimmed_messages)))
    return trimmed_messages


def get_modified_chat_history_for_image_input(user_message: ChatMessage,
                                              chat_history: list) -> list:
    '''
    Add requesting message for image
    '''

    image_url_replacement_question = 'Is the fashion item in this image available?'

    if not user_message.is_image:
        return chat_history

    for i in range(len(chat_history) - 1, -1, -1):
        if chat_history[i]['role'] == 'user':
            chat_history[i]['content'] = image_url_replacement_question
            break

    return chat_history


def get_separate_message_histories(rows: QuerySet) -> tuple[list[dict], list[dict]]:
    '''
    Gets separate chat and retrieval assistant message histories
    '''

    chat_message_history, retrieval_message_history = [], []
    retrieval_assistant_message_count, chat_assistant_message_count = 0, 0
    user_message_count = 0

    for row in rows:
        message_dict = {'role': row.agent.lower(), 'content': row.message}

        if row.agent == Agent.USER:
            if user_message_count < (CHAT_HISTORY_LIMIT + 1):
                retrieval_message_history.append(message_dict)
                chat_message_history.append(message_dict)
                user_message_count += 1

        elif row.agent == Agent.ASSISTANT:
            if row.agent_subtype == AgentSubType.RETRIEVAL_ASSISTANT:
                if retrieval_assistant_message_count < RETRIEVAL_HISTORY_LIMIT:
                    retrieval_message_history.append(message_dict)
                    retrieval_assistant_message_count += 1
            elif row.agent_subtype == AgentSubType.CHAT_ASSISTANT:
                if chat_assistant_message_count < CHAT_HISTORY_LIMIT:
                    chat_message_history.append(message_dict)
                    chat_assistant_message_count += 1

        if user_message_count >= (CHAT_HISTORY_LIMIT + 1) and \
                retrieval_assistant_message_count >= RETRIEVAL_HISTORY_LIMIT and \
                chat_assistant_message_count >= CHAT_HISTORY_LIMIT:
            break

    return (retrieval_message_history[::-1], chat_message_history[::-1])


def get_timestamp_intervals(user_message: ChatMessage) -> tuple[str, str]:
    '''
    Gets the timestamp intervals for fetching chat history
    '''

    user_message_timestamp = timezone.make_aware(user_message.timestamp) if timezone.is_naive(
        user_message.timestamp) else user_message.timestamp
    previous_timestamp = user_message_timestamp - \
        timedelta(seconds=RETRIEVAL_HISTORY_DURATION)

    return previous_timestamp, user_message_timestamp


def get_user_message(request_data: dict) -> ChatMessage:
    '''
    Creates chat message based on user's chat message request
    '''

    is_image = False
    image_url = None

    match = re.search(URL_PATTERN, request_data.get('message', ''))

    if match:
        is_image = True
        image_url = match.group(0)

    user_message = ChatMessage(
        user_id=request_data.get('userId'),
        session_id=request_data.get('sessionId'),
        request_id=request_data.get('requestId'),
        agent=Agent.USER,
        agent_subtype=None,
        timestamp=timezone.now(),
        model=None,
        message=request_data.get('message'),
        is_image=is_image,
        image_url=image_url)

    return user_message
