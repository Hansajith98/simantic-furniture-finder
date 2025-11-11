from django.db import models
from django.contrib.postgres import fields
import uuid


class ProcessState(models.TextChoices):
    WAITING = 'WAITING'
    PROCESSING = 'PROCESSING'
    COMPLETED = 'COMPLETED'
    FAILED = 'FAILED'


class KnowledgeBaseTypes(models.TextChoices):
    DOCUMENT = 'DOCUMENT'
    WEB_SCRAPE = 'WEB_SCRAPE'


class UsedFunctionName(models.TextChoices):
    IMAGE_SUMMARY_CREATION = 'IMAGE_SUMMARY_CREATION'
    WEB_PAGE_SUMMARY_CREATION = 'WEB_PAGE_SUMMARY_CREATION'
    EMBEDDING_CREATION = 'EMBEDDING_CREATION'
    USER_MESSAGE_REPHRASER = 'USER_MESSAGE_REPHRASER'
    ASSISTANT_MESSAGE_GENERATOR = 'ASSISTANT_MESSAGE_GENERATOR'


class PromptType(models.TextChoices):
    CHAT_PROMPT = 'CHAT_PROMPT'
    WEB_SUMMARY_PROMPT = 'WEB_SUMMARY_PROMPT'
    IMAGE_SUMMARY_PROMPT = 'IMAGE_SUMMARY_PROMPT'


class GeneralInfo(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    general_information = models.JSONField(null=True, blank=True, default=dict)
    vectorization_status = models.TextField(
        max_length=50,
        choices=ProcessState.choices,
        default=ProcessState.WAITING)
    vector_id = models.CharField(max_length=255, blank=True, null=True)
    vector = fields.ArrayField(models.FloatField(), blank=True, null=True)
    update_on_vector_db_status = models.CharField(
        max_length=50,
        choices=ProcessState.choices,
        default=ProcessState.WAITING)
    is_delete = models.BooleanField(default=False, null=False, blank=False)
    created_on = models.DateTimeField(auto_now_add=True)
    updated_on = models.DateTimeField(auto_now=True)


class Organization(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    owner = models.CharField(max_length=255)
    start_url = models.URLField(max_length=255)
    allowed_domains = models.TextField()
    general_info = models.ForeignKey(
        GeneralInfo, on_delete=models.CASCADE, null=True,
        related_name="organization_general_info")
    site_sync_status = models.CharField(
        max_length=50,
        choices=ProcessState.choices,
        default=ProcessState.WAITING)
    site_scraper_status = models.CharField(
        max_length=50,
        choices=ProcessState.choices,
        default=ProcessState.WAITING)
    site_embedding_status = models.CharField(
        max_length=50,
        choices=ProcessState.choices,
        default=ProcessState.WAITING)
    is_delete = models.BooleanField(default=False, null=False, blank=False)
    created_on = models.DateTimeField(auto_now_add=True)
    updated_on = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'organization'

    def __str__(self):
        return f"{self.name}"


class ProductWebPage(models.Model):
    id = models.CharField(max_length=255, primary_key=True, editable=False)
    organization = models.ForeignKey(
        Organization, on_delete=models.CASCADE, related_name="product_webpages")
    url = models.URLField(unique=False)
    page_title = models.CharField(max_length=255, blank=True, null=True)
    page_markdown = models.TextField(blank=True, null=True)
    page_summary = models.JSONField(blank=True, null=True)
    vector_id = models.CharField(max_length=255, blank=True, null=True)
    vector = fields.ArrayField(models.FloatField(), blank=True, null=True)
    thumbnail_url = models.URLField(blank=True, null=True)
    web_content_hash = models.CharField(max_length=64, blank=True, null=True)
    scrape_status = models.CharField(
        max_length=50,
        choices=ProcessState.choices,
        default=ProcessState.WAITING)
    summary_creation_status = models.CharField(
        max_length=50,
        choices=ProcessState.choices,
        default=ProcessState.WAITING)
    vectorization_status = models.CharField(
        max_length=50,
        choices=ProcessState.choices,
        default=ProcessState.WAITING)
    update_on_vector_db_status = models.CharField(
        max_length=50,
        choices=ProcessState.choices,
        default=ProcessState.WAITING)
    scraped_on = models.DateTimeField(blank=True, null=True)
    is_delete = models.BooleanField(default=False, null=False, blank=False)
    created_on = models.DateTimeField(auto_now_add=True)
    updated_on = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'product_webpage'
        unique_together = ('organization', 'url')

    def __str__(self):
        return f"{self.organization.name}-Product Web Pages"


class ProductImage(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    product_webpage = models.ForeignKey(
        ProductWebPage, on_delete=models.CASCADE, related_name="product_images")
    url = models.URLField()
    image_description = models.JSONField(blank=True, null=True)
    vector_id = models.CharField(max_length=255, blank=True, null=True)
    vector = fields.ArrayField(models.FloatField(), blank=True, null=True)
    summary_creation_status = models.CharField(
        max_length=50,
        choices=ProcessState.choices,
        default=ProcessState.WAITING)
    vectorization_status = models.CharField(
        max_length=50,
        choices=ProcessState.choices,
        default=ProcessState.WAITING)
    update_on_vector_db_status = models.CharField(
        max_length=50,
        choices=ProcessState.choices,
        default=ProcessState.WAITING)
    scraped_on = models.DateTimeField(blank=True, null=True)
    is_delete = models.BooleanField(default=False, null=False, blank=False)

    created_on = models.DateTimeField(auto_now_add=True)
    updated_on = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'product_image'
        unique_together = ('product_webpage', 'url')

    def __str__(self):
        return f"{self.product_webpage.organization.name}-Product Image"


class Configuration(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    organization = models.ForeignKey(
        Organization, on_delete=models.CASCADE, related_name="configurations")
    sitemap_configuration = models.JSONField(default=dict)
    scrape_configuration = models.JSONField(default=dict)
    image_summary_required = models.BooleanField(default=False)
    knowledge_base_type = models.CharField(
        max_length=50,
        choices=KnowledgeBaseTypes.choices,
        default=KnowledgeBaseTypes.WEB_SCRAPE)
    additional_configurations = models.JSONField(default=dict)
    is_delete = models.BooleanField(default=False, null=False, blank=False)

    created_on = models.DateTimeField(auto_now_add=True)
    updated_on = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'configuration'
        indexes = [
            models.Index(fields=['id']),
            models.Index(fields=['organization']),
        ]

    def __str__(self):
        return f"{self.organization.name}-Configurations"


class TokenUsage(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    organization = models.ForeignKey(
        Organization, on_delete=models.CASCADE, related_name="token_usage")
    request_id = models.CharField(max_length=255, blank=True, null=True)
    input_tokens = models.IntegerField(default=0)
    completion_tokens = models.IntegerField(default=0)
    total_tokens = models.IntegerField(default=0)
    model = models.CharField(max_length=255, blank=True, null=True)
    used_function_name = models.CharField(
        max_length=50,
        choices=UsedFunctionName.choices)
    is_delete = models.BooleanField(default=False, null=False, blank=False)

    created_on = models.DateTimeField(auto_now_add=True)
    updated_on = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'token_usage'
        indexes = [
            models.Index(fields=['id']),
            models.Index(fields=['organization']),
        ]

    def __str__(self):
        return f"{self.organization.name}-Token Usage"


class Prompt(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE,
                                     related_name="prompts")
    webpage_summary_prompt = models.TextField(blank=True, null=True)
    vector_rephrase_prompt = models.TextField(blank=True, null=True)
    chat_response_prompt = models.TextField(blank=True, null=True)
    graph_retrieval_prompt = models.TextField(blank=True, null=True)
    image_desc_prompt = models.TextField(blank=True, null=True)
    is_delete = models.BooleanField(default=False, null=False, blank=False)

    created_on = models.DateTimeField(auto_now_add=True)
    updated_on = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'prompt'
        indexes = [
            models.Index(fields=['id']),
        ]

    def __str__(self):
        return f"{self.organization.name}-Prompts"


class ProductCollection(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    url = models.URLField()
    name = models.CharField(max_length=255, blank=True, null=True)
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE,
                                     related_name="product_collections")
    is_delete = models.BooleanField(default=False, null=False, blank=False)

    created_on = models.DateTimeField(auto_now_add=True)
    updated_on = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'product_collection'

    def __str__(self):
        return f"{self.organization.name}-Product Collection"


class MappingProductCollection(models.Model):
    product = models.ForeignKey(
        ProductWebPage, on_delete=models.CASCADE, related_name="product_webpage")
    collection = models.ForeignKey(
        ProductCollection, on_delete=models.CASCADE,
        related_name="product_collection")
    date_joined = models.DateTimeField(auto_now_add=True)
    updated_on = models.DateTimeField(auto_now=True)
    is_delete = models.BooleanField(default=False, null=False, blank=False)

    class Meta:
        db_table = 'mapping_product_collection'
        unique_together = ['product', 'collection']

    def __str__(self):
        return f"{self.product.organization.name}-Mapping Product Collection"


class PromptTool(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    prompt_type = models.CharField(
        max_length=50,
        choices=PromptType.choices,
        default=PromptType.CHAT_PROMPT)
    tools = models.JSONField(blank=False, default=list)
    prompt = models.ForeignKey(Prompt, on_delete=models.CASCADE,
                               related_name="prompt_tools")
    is_delete = models.BooleanField(default=False, null=False, blank=False)

    created_on = models.DateTimeField(auto_now_add=True)
    updated_on = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'prompt_tool'

    def __str__(self):
        return f"{self.prompt.organization.name}-Prompt Tools"
