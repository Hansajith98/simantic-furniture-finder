from django.db import models
from organization.models import Organization
import uuid


class Agent(models.TextChoices):
    USER = 'user'
    ASSISTANT = 'assistant'


class AgentSubType(models.TextChoices):
    CHAT_ASSISTANT = 'chat_assistant'
    RETRIEVAL_ASSISTANT = 'retrieval_assistant'
    TOOL_ASSISTANT = 'tool_assistant'


class ChatMessage(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user_id = models.CharField(max_length=255, blank=True, null=True)
    session_id = models.CharField(max_length=255, blank=True, null=True)
    request_id = models.CharField(max_length=255, blank=True, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    message = models.TextField(default='Null')
    agent = models.CharField(
        max_length=10, choices=Agent.choices, default=Agent.USER)
    agent_subtype = models.CharField(
        max_length=20, choices=AgentSubType.choices, null=True, blank=True)
    is_image = models.CharField(max_length=255, blank=True, null=True)
    image_url = models.CharField(max_length=255, blank=True, null=True)
    model = models.CharField(max_length=255, blank=True, null=True)
    prompt_tokens = models.IntegerField(default=0)
    completion_tokens = models.IntegerField(default=0)
    total_tokens = models.IntegerField(default=0)
    is_delete = models.BooleanField(default=False, null=False, blank=False)

    created_on = models.DateTimeField(auto_now_add=True)
    updated_on = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'chat_message'
        indexes = [
            models.Index(fields=['id']),
        ]

    # def __str__(self):
    #     return f"{self.role}: {self.message[:20]}..."
