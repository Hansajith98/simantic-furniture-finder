from django.db import models
from organization.models import Organization
import uuid


class Role(models.TextChoices):
    USER = 'user'
    ASSISTANT = 'assistant'


class AssistantType(models.TextChoices):
    CHAT = 'chat_assistant'
    RETRIEVAL = 'retrieval_assistant'
    TOOL = 'tool_assistant'


class ConversationMessage(models.Model):
    uid = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user_identifier = models.CharField(max_length=255, null=True, blank=True)
    session_identifier = models.CharField(max_length=255, null=True, blank=True)
    request_identifier = models.CharField(max_length=255, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    body = models.TextField(default='Null')
    role = models.CharField(max_length=20, choices=Role.choices, default=Role.USER)
    role_subtype = models.CharField(
        max_length=25, choices=AssistantType.choices, blank=True, null=True
    )
    contains_image = models.CharField(max_length=255, blank=True, null=True)
    image_source = models.CharField(max_length=255, blank=True, null=True)
    model_name = models.CharField(max_length=255, blank=True, null=True)
    prompt_token_count = models.IntegerField(default=0)
    completion_token_count = models.IntegerField(default=0)
    total_token_count = models.IntegerField(default=0)
    marked_deleted = models.BooleanField(default=False)

    created_on = models.DateTimeField(auto_now_add=True)
    modified_on = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'chat_message'
        indexes = [
            models.Index(fields=['uid']),
        ]

    def __str__(self):
        return f"{self.role}: {self.body[:30]}..."
