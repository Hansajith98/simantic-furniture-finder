import json
import re
import logging
from datetime import timedelta
from configparser import ConfigParser
import tiktoken

from django.db import transaction
from django.db.models import QuerySet
from django.utils import timezone
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from chatter.models import ChatMessage, Agent, AgentSubType
from chatter.utils import get_response
from organization.models import TokenUsage
from organization.utils import log_execution_time


log = logging.getLogger("semantic_furniture_finder")

URL_REGEX = r"(https?://(?:www\.)?[\w-]+(?:\.[\w-]+)+(?:/[\w\-./?%&=]*)?)"

cfg = ConfigParser()
cfg.read("conf/application.ini")

HISTORY_CHAT_LIMIT = int(cfg["Conversation"]["chat_history_limit"])
HISTORY_RETRIEVE_LIMIT = int(cfg["Conversation"]["retrieval_history_limit"])
HISTORY_DURATION_SEC = int(cfg["Conversation"]["history_duration"])
TOKEN_LIMIT = int(cfg["OpenAI"]["message_input_max_tokens"])

MAX_HISTORY_COUNT = max(HISTORY_CHAT_LIMIT, HISTORY_RETRIEVE_LIMIT)


class TokenLimitExceeded(Exception):
    def __init__(self):
        super().__init__("Message token count exceeded.")


class ChatSessionView(APIView):
    @transaction.atomic
    def post(self, request):
        try:
            data = request.data
            log.info(f"chat_request_incoming - [{data}]")

            if not data:
                return Response(
                    {"message": "Empty request body."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            needed_fields = ["message", "userId", "sessionId", "requestId"]
            if not all(f in data for f in needed_fields):
                return Response(
                    {"message": "Missing required fields."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            user_msg = create_user_message(data)
            user_msg.save()

            chat_history = assemble_conversation_history(user_msg)

            assistant_msg = get_response(user_msg, chat_history)
            assistant_msg.save()

            log.info(
                f"[{user_msg.request_id}] - [{user_msg.session_id}] - chat_message_stored"
            )

            token_entries = TokenUsage.objects.filter(
                request_id=data.get("requestId"), is_delete=False
            )

            tokens_data = [
                {
                    "inputToken": t.input_tokens,
                    "outputToken": t.completion_tokens,
                    "totalToken": t.total_tokens,
                    "model": t.model,
                    "function": t.used_function_name,
                }
                for t in token_entries
            ]

            reply_text = parse_chat_reply(assistant_msg.message)

            return Response(
                {
                    "userId": data["userId"],
                    "sessionId": data["sessionId"],
                    "content": {
                        "message": reply_text,
                        "suggestions": "",
                        "reference": "",
                        "tokenInfo": tokens_data,
                    },
                },
                status=status.HTTP_201_CREATED,
            )

        except TokenLimitExceeded:
            transaction.set_rollback(True)
            log.error("token_limit_exceeded", exc_info=True)
            return Response(
                {"message": "Message Token Count Exceeded."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        except Exception:
            transaction.set_rollback(True)
            log.error("internal_server_failure", exc_info=True)
            return Response(
                {"message": "Internal Server Error"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


def parse_chat_reply(msg_obj):
    content_json = json.loads(msg_obj.get("content")[0].get("text"))
    return content_json.get("chatReply")


@log_execution_time
def assemble_conversation_history(user_msg: ChatMessage) -> dict:
    start_ts, end_ts = calculate_time_interval(user_msg)
    qs = (
        ChatMessage.objects.filter(
            user_id=user_msg.user_id,
            session_id=user_msg.session_id,
            rag_type=user_msg.rag_type,
            timestamp__range=(start_ts, end_ts),
        )
        .order_by("-timestamp")
    )

    retrieval_msgs, chat_msgs = divide_histories(qs)

    log.info(f"retrieval_history_len={len(retrieval_msgs)}")
    log.info(f"chat_history_len={len(chat_msgs)}")

    chat_msgs = adjust_for_image_input(user_msg, chat_msgs)
    chat_msgs = trim_history_within_token_limit(chat_msgs)

    if not chat_msgs:
        raise TokenLimitExceeded

    retrieval_msgs = retrieval_msgs[-len(chat_msgs) :]
    log.info(
        f"conversation_loaded - retr:{len(retrieval_msgs)} chat:{len(chat_msgs)}"
    )

    return {"retrieval": retrieval_msgs, "chat": chat_msgs}


def trim_history_within_token_limit(messages, limit=TOKEN_LIMIT):
    encoder = tiktoken.encoding_for_model("gpt-4o-mini")
    count = len(encoder.encode("".join(m["content"] for m in messages)))
    reduced = messages

    while count > limit and reduced:
        reduced = reduced[1:]
        count = len(encoder.encode("".join(m["content"] for m in reduced)))
    return reduced


def adjust_for_image_input(user_msg: ChatMessage, history: list) -> list:
    image_query = "Is the fashion item in this image available?"
    if not user_msg.is_image:
        return history

    for idx in range(len(history) - 1, -1, -1):
        if history[idx]["role"] == "user":
            history[idx]["content"] = image_query
            break
    return history


def divide_histories(rows: QuerySet):
    chat_hist, retrieval_hist = [], []
    chat_assistant_count = retr_assistant_count = user_count = 0

    for r in rows:
        msg = {"role": r.agent.lower(), "content": r.message}

        if r.agent == Agent.USER:
            if user_count < HISTORY_CHAT_LIMIT + 1:
                retrieval_hist.append(msg)
                chat_hist.append(msg)
                user_count += 1

        elif r.agent == Agent.ASSISTANT:
            if r.agent_subtype == AgentSubType.RETRIEVAL_ASSISTANT:
                if retr_assistant_count < HISTORY_RETRIEVE_LIMIT:
                    retrieval_hist.append(msg)
                    retr_assistant_count += 1
            elif r.agent_subtype == AgentSubType.CHAT_ASSISTANT:
                if chat_assistant_count < HISTORY_CHAT_LIMIT:
                    chat_hist.append(msg)
                    chat_assistant_count += 1

        if (
            user_count >= HISTORY_CHAT_LIMIT + 1
            and retr_assistant_count >= HISTORY_RETRIEVE_LIMIT
            and chat_assistant_count >= HISTORY_CHAT_LIMIT
        ):
            break

    return retrieval_hist[::-1], chat_hist[::-1]


def calculate_time_interval(user_msg: ChatMessage):
    msg_time = (
        timezone.make_aware(user_msg.timestamp)
        if timezone.is_naive(user_msg.timestamp)
        else user_msg.timestamp
    )
    start = msg_time - timedelta(seconds=HISTORY_DURATION_SEC)
    return start, msg_time


def create_user_message(data: dict) -> ChatMessage:
    has_image = False
    img_url = None

    url_match = re.search(URL_REGEX, data.get("message", ""))
    if url_match:
        has_image = True
        img_url = url_match.group(0)

    return ChatMessage(
        user_id=data.get("userId"),
        session_id=data.get("sessionId"),
        request_id=data.get("requestId"),
        agent=Agent.USER,
        agent_subtype=None,
        timestamp=timezone.now(),
        model=None,
        message=data.get("message"),
        is_image=has_image,
        image_url=img_url,
    )
