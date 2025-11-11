import json
import logging

from rest_framework.views import APIView
from django.db import transaction
from rest_framework.response import Response
from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.exceptions import APIException


from external_services.weaviate_services import WeaviateServices, WeaviateServicesForDocument
from external_services.neo4j_services import Neo4jServices
from organization.models import *
from chat.models import ChatMessage, Agent
from organization.serializers import *
from data_processor.tasks import scrape_sitemap
from utils.response import *


logger = logging.getLogger('chat_with_website')
logger.setLevel(logging.DEBUG)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class TestView(APIView):
    def get(self, request):
        return Response({"message": "pong"}, status=status.HTTP_200_OK)


class OrganizationView(APIView):

    def get(self, request: Request, id: str) -> Response:
        try:
            organization = Organization.objects.get(id=id)
            webpages_count = ProductWebPage.objects.filter(
                organization=organization).count()
            scraped_on = ProductWebPage.objects.filter(
                organization=organization).latest('scraped_on').scraped_on
            return Response(data={
                "organizationId": organization.id,
                "numberOfWebPages": webpages_count,
                "createdOn": organization.created_on,
                "lastScrappedOn": scraped_on
            }, status=status.HTTP_200_OK)

        except Organization.DoesNotExist:
            transaction.set_rollback(True)
            response = get_resource_not_exists_response(
                resource="organization")
            logger.info(f"{response.message} {response.status_code}")
            return response.response

        except Exception as ex:
            transaction.set_rollback(True)
            logger.exception(
                f"Error occurred during organization retrieval: {str(ex)}")
            response = get_server_error_response('Error getting organization')
            return response.response

    @transaction.atomic
    def post(self, request: Request) -> Response:
        try:
            request_data = request.data
            if not request_data:
                response = get_invalid_request_response()
                logger.info(f"{response.message} {response.status_code}")
                return response.response

            knowledge_base_type = request_data.get(
                'knowledgeBaseType', 'none').upper()

            match knowledge_base_type:
                case KnowledgeBaseTypes.DOCUMENT.value:
                    return self.create_document_type_organization(request_data)
                case KnowledgeBaseTypes.WEB_SCRAPE.value:
                    return self.create_web_scrape_type_organization(request_data)
                case _:
                    return get_invalid_request_response('invalid knowledgeBaseType').response

        except Exception:
            transaction.set_rollback(True)
            response = get_server_error_response(
                'organization creation error!')
            logger.info(
                f"{response.message} {response.status_code}", exc_info=True)
            return response.response

    @transaction.atomic
    def delete(self, request: Request) -> Response:
        try:
            organization_id = request.query_params.get('organizationId')
            if not organization_id:
                return Response({"message": "organization id not provided"},
                                status=status.HTTP_404_NOT_FOUND)

            organization = Organization.objects.get(id=organization_id)
            configuration = Configuration.objects.get(
                organization=organization, is_delete=False)

            if configuration.knowledge_base_type == KnowledgeBaseTypes.DOCUMENT:
                response = self.delete_document_type_organization(
                    organization)
            else:
                response = self.delete_web_scrape_type_organization(
                    organization)

            return response

        except Configuration.DoesNotExist:
            transaction.set_rollback(True)
            response = get_resource_not_exists_response(
                resource="Configuration")
            logger.info(f"{response.message} {response.status_code}")
            return response.response

        except Organization.DoesNotExist:
            transaction.set_rollback(True)
            response = get_resource_not_exists_response(
                resource="organization")
            logger.info(f"{response.message} {response.status_code}")
            return response.response

        except Exception:
            transaction.set_rollback(True)
            response = get_server_error_response(
                message="organization deletion error!")
            logger.error(
                f"{response.message} {response.status_code}", exc_info=True)
            return response.response

    def delete_document_type_organization(self, organization: Organization) \
            -> Response:
        '''
        Delete organization of document type
        '''
        try:
            weaviate_services = WeaviateServicesForDocument()

            Configuration.objects.filter(organization=organization).delete()
            if organization.general_info:
                organization.general_info.delete()
            weaviate_services.delete_weaviate_collection(organization.name)
            organization.delete()

            response = get_organization_deleted_response()
            logger.info(f"{response.message} {response.status_code}")
            return response.response

        except Exception:
            response = get_server_error_response(
                message="organization deletion error!")
            logger.error(
                f"{response.message} {response.status_code}", exc_info=True)
            return response.response

    def delete_web_scrape_type_organization(self, organization: Organization) \
            -> Response:
        '''
        Delete organization of web scrape type
        '''
        try:
            neo4j_services = Neo4jServices(logger.name)
            weaviate_services = WeaviateServices()

            MappingProductCollection.objects.filter(
                collection__organization=organization).delete()
            ProductWebPage.objects.filter(
                organization=organization).delete()
            ProductCollection.objects.filter(
                organization=organization).delete()
            Configuration.objects.filter(organization=organization).delete()

            neo4j_services.delete_organization(str(organization.id))
            neo4j_services.commit()

            if organization.general_info:
                organization.general_info.delete()
            weaviate_services.delete_weaviate_collection(organization.name)
            organization.delete()

            response = get_organization_deleted_response()
            logger.info(f"{response.message} {response.status_code}")
            return response.response

        except Exception:
            response = get_server_error_response(
                message="organization deletion error!")
            if neo4j_services:
                neo4j_services.rollback()
            logger.error(
                f"{response.message} {response.status_code}", exc_info=True)
            return response.response
        finally:
            if neo4j_services:
                neo4j_services.close()

    def create_document_type_organization(self, request_data: dict) -> Response:
        '''
        Create organization of document type
        '''
        try:
            weaviate_services = WeaviateServicesForDocument()

            required_fields = ["id", "name", "owner", "knowledgeBaseType"]

            if not all(item in request_data for item in required_fields):
                response = get_invalid_request_response(
                    message="required field incomplete")
                logger.info(f"{response.message} {response.status_code}")
                return response.response

            organization_id = request_data.get('id')

            (organization, created) = Organization.objects.get_or_create(
                id=organization_id,
                name=request_data["name"],
                owner=request_data["owner"])

            if not created:
                response = get_organization_exists_response()
                logger.info(f"{response.message} {response.status_code}")
                return response.response

            (configuration, created) = Configuration.objects.get_or_create(
                organization_id=organization_id,
                knowledge_base_type=KnowledgeBaseTypes.DOCUMENT,
                additional_configurations=request_data.get(
                    'additionalConfigurations', {}))

            weaviate_services.create_weaviate_collection(request_data["name"])

            response = get_resource_created_response(resource="organization")
            logger.info(f"{response.message} {response.status_code}")
            return response.response

        except Exception:
            transaction.set_rollback(True)
            if weaviate_services:
                weaviate_services.delete_weaviate_collection(
                    request_data["name"])

            response = get_server_error_response(
                message="organization creation error!")
            logger.error(
                f"{response.message} {response.status_code}", exc_info=True)
            return response.response

    def create_web_scrape_type_organization(self, request_data: dict) \
            -> Response:
        '''
        Create organization of web scrape type
        '''
        try:
            logger.info(
                f"Create organization request received [{request_data}]")
            neo4j_services = Neo4jServices(logger.name)
            weaviate_services = WeaviateServices()

            required_fields = ["id", "name", "owner", "allowedDomains",
                               "startUrl", "spiderConfiguration",
                               "imageSummaryRequired"]

            if not all(item in request_data for item in required_fields):
                response = get_invalid_request_response(
                    message="required field incomplete")
                logger.info(f"{response.message} {response.status_code}")
                return response.response

            organization_id = request_data.get('id')

            logger.info(f"Start creating organization [{organization_id}]")

            spider_configuration = request_data.get('spiderConfiguration')

            allowed_domains = request_data.get('allowedDomains')
            start_url = request_data.get('startUrl')

            (organization, created) = Organization.objects.get_or_create(
                id=organization_id,
                name=request_data["name"],
                owner=request_data["owner"],
                allowed_domains=allowed_domains,
                start_url=start_url
            )

            if not created:
                response = get_organization_exists_response()
                logger.info(f"{response.message} {response.status_code}")
                return response.response

            if (general_info := request_data.get('generalInfo')):
                general_info = GeneralInfo.objects.create(
                    general_information=general_info
                )

            organization.general_info = general_info if general_info else None
            organization.save()

            logger.info(
                f"Organization entry created for [[{organization_id}]]")

            Configuration.objects.create(
                organization_id=organization_id,
                sitemap_configuration=spider_configuration.get(
                    'sitemapConfiguration'),
                scrape_configuration=spider_configuration.get(
                    'scrapeConfiguration'),
                image_summary_required=request_data.get(
                    'imageSummaryRequired'),
                additional_configurations=request_data.get(
                    'additionalConfigurations', {}))

            logger.info(
                f"Configuration entry created for [[{organization_id}]]")

            default_collection = ProductCollection(
                url='default//:default-collection',
                name='Default Collection',
                organization_id=organization_id
            )
            default_collection.save()

            logger.info(f"Product collection created [[{organization_id}]]")

            weaviate_services.create_weaviate_collection(request_data["name"])

            logger.info(f"Weaviate collection created [[{organization_id}]]")

            neo4j_services.create_organization(organization_id)
            neo4j_services.insert_collection({
                "organizationId": str(default_collection.organization.id),
                "name": default_collection.name,
                "url": default_collection.url
            })
            neo4j_services.commit()

            logger.info(f"Neo4j collection created [[{organization_id}]]")

            # Uncomment below line to enable sitemap spider automatically when organization is created
            # scrape_sitemap.delay_on_commit(organization_id)
            # logger.info(f"Request sent for sitemap spider [{organization_id}]")

            response = get_resource_created_response(resource="organization")
            logger.info(f"{response.message} {response.status_code}")
            return response.response
        except Exception as ex:
            logger.error(
                f"Error creating organization [[{organization_id}]] [{str(ex)}]")
            transaction.set_rollback(True)
            if neo4j_services:
                neo4j_services.rollback()
            if weaviate_services:
                weaviate_services.delete_weaviate_collection(
                    request_data["name"])

            response = get_server_error_response(
                message="organization creation error!")
            logger.error(
                f"{response.message} {response.status_code}", exc_info=True)
            return response.response
        finally:
            if neo4j_services:
                neo4j_services.close()


class GeneralInfoView(APIView):
    def get(self, request: Request, *args, **kwargs) -> Response:
        organization_id = request.query_params.get('organization')
        logger.info(f"Fetching general info - [{organization_id}]")

        serializer = GeneralInfoSerializer(
            data=request.data,
            context={"organization_id": organization_id}
        )
        if serializer.is_valid():
            general_info = serializer.save()
            data = serializer.to_representation(general_info)
            logger.info(
                f"general_info fetched - [{organization_id}] - {general_info.id}")
            return Response(data=data, status=status.HTTP_200_OK)

        response = get_invalid_request_response(data=serializer.errors)
        logger.info(f"{response.message} {response.status_code}")
        return response.response

    @transaction.atomic
    def post(self, request: Request) -> Response:
        organization_id = request.data.get("organization_id")

        serializer = GeneralInfoSerializer(
            data=request.data,
            context={"organization_id": organization_id}
        )

        if serializer.is_valid():
            general_info = serializer.save()
            response = get_resource_created_response(resource="general info")
            logger.info(f"{response.message} {response.status_code}")
            return response.response

        response = get_invalid_request_response(data=serializer.errors)
        logger.info(f"{response.message} {response.status_code}")
        return response.response

    @transaction.atomic
    def put(self, request: Request) -> Response:
        organization_id = request.data.get("organization_id")

        serializer = GeneralInfoSerializer(
            data=request.data,
            context={"organization_id": organization_id},
            partial=True
        )

        if serializer.is_valid():
            general_info = serializer.save()
            response = get_resource_updated_response(resource="general info")
            logger.info(f"{response.message} {response.status_code}")
            return response.response

        response = get_invalid_request_response(data=serializer.errors)
        logger.info(f"{response.message} {response.status_code}")
        return response.response


class PromptView(APIView):
    @transaction.atomic
    def post(self, request: Request) -> Response:
        try:
            request_data = request.data
            if not request_data:
                response = get_invalid_request_response()
                logger.info(f"{response.message} {response.status_code}")
                return response.response

            required_fields = ['organizationId', 'vectorRephrasePrompt',
                               'graphRetrievalPrompt', 'webpageSummaryPrompt',
                               'chatResponsePrompt']

            if not all(item in request_data for item in required_fields):
                logger.info(request_data.keys())
                response = get_invalid_request_response(
                    message="required fields incomplete!")
                logger.info(f"{response.message} {response.status_code}")
                return response.response

            if not Organization.objects.filter(id=request_data['organizationId']).exists():
                response = get_resource_not_exists_response('Organization')
                logger.info(f"{response.message} {response.status_code}")
                return response.response

            if Prompt.objects.filter(organization_id=request_data[
                    "organizationId"], is_delete=False).exists():
                response = get_prompt_already_created_response()
                logger.info(f"{response.message} {response.status_code}")
                return response.response

            prompt = Prompt(
                organization_id=request_data.get("organizationId"),
                webpage_summary_prompt=request_data.get(
                    "webpageSummaryPrompt"),
                vector_rephrase_prompt=request_data.get(
                    "vectorRephrasePrompt"),
                chat_response_prompt=request_data.get("chatResponsePrompt"),
                graph_retrieval_prompt=request_data.get(
                    "graphRetrievalPrompt"),
                image_desc_prompt=request_data.get("imageDescPrompt")
            )
            prompt.save()

            if (tools := request_data.get('tools')):
                tools = json.loads(tools)
                for i in tools:
                    prompt_type_str = i.get('prompt_type')
                    match prompt_type_str:
                        case 'web_summary_prompt':
                            prompt_type = PromptType.WEB_SUMMARY_PROMPT
                        case 'image_summary_prompt':
                            prompt_type = PromptType.IMAGE_SUMMARY_PROMPT
                        case 'chat_prompt':
                            prompt_type = PromptType.CHAT_PROMPT
                        case _:
                            logger.info(
                                f"invalid_prompt_type_received - {prompt_type_str}")
                            continue
                    PromptTool.objects.create(
                        prompt=prompt,
                        prompt_type=prompt_type,
                        tools=i.get('tools', [])
                    )
                    logger.info(
                        f"updated_prompt_tools [{prompt.organization_id}]")

            response = get_resource_created_response(resource="prompts")
            logger.info(
                f"{response.message} {response.status_code} [{prompt.organization_id}]")
            return response.response

        except:
            transaction.set_rollback(True)
            response = get_server_error_response(
                message="prompts creation error!")
            logger.error(
                f"{response.message} {response.status_code}", exc_info=True)
            return response.response

    def put(self, request: Request, *args, **kwargs) -> Response:
        try:

            request_data = request.data
            if not request_data:
                response = get_invalid_request_response()
                logger.info(f"{response.message} {response.status_code}")
                return response.response

            if "organizationId" not in request_data:
                response = get_invalid_request_response(
                    message="organization id not provided")
                logger.info(f"{response.message} {response.status_code}")
                return response.response

            try:
                prompt = Prompt.objects.get(
                    organization_id=request_data["organizationId"], is_delete=False)
            except Prompt.DoesNotExist:
                response = get_resource_not_exists_response(resource="prompt")
                logger.info(f"{response.message} {response.status_code}")
                return response.response

            if (webpage_summary_prompt := request_data.get("webpageSummaryPrompt")):
                prompt.webpage_summary_prompt = webpage_summary_prompt

            if (vector_rephrase_prompt := request_data.get("vectorRephrasePrompt")):
                prompt.vector_rephrase_prompt = vector_rephrase_prompt

            if (chat_response_prompt := request_data.get("chatResponsePrompt")):
                prompt.chat_response_prompt = chat_response_prompt

            if (graph_retrieval_prompt := request_data.get("graphRetrievalPrompt")):
                prompt.graph_retrieval_prompt = graph_retrieval_prompt

            if (image_desc_prompt := request_data.get("imageDescPrompt")):
                prompt.image_desc_prompt = image_desc_prompt
            prompt.save()

            if (tools := request_data.get('tools')):
                tools = json.loads(tools)
                for i in tools:
                    if (prompt_type := i.get('prompt_type')) not in [
                            'web_summary_prompt', 'image_summary_prompt',
                            'chat_prompt']:
                        logger.info(
                            f"invalid_prompt_type_received - {prompt_type}")
                    if (tools_obj := PromptTool.objects.filter(
                            prompt=prompt, prompt_type=prompt_type,
                            is_delete=False).first()):
                        tools_obj.tools = i.get('tools', [])
                        tools_obj.save()
                    else:
                        PromptTool.objects.create(
                            prompt=prompt,
                            prompt_type=prompt_type,
                            tools=i.get('tools', [])
                        )
                    logger.info(
                        f"updated_prompt_tools [{prompt.organization_id}]")

            response = get_prompt_updated_response()
            logger.info(f"{response.message} {response.status_code}")
            return response.response

        except:
            response = get_server_error_response(
                message="organization update error")
            logger.error(
                f"{response.message} {response.status_code}", exc_info=True)
            return response.response


class OrganizationStatInfoView(APIView):
    def get(self, request: Request, *args, **kwargs) -> Response:
        organization_id = request.query_params.get("organization")
        if not organization_id:
            response = get_invalid_request_response()
            logger.info(f"{response.message} {response.status_code}")
            return response.response
        statistics_data = TokenUsage.objects.filter(
            organization_id=organization_id, is_delete=False)

        message_input_token_count = sum(statistics_data.filter(
            used_function_name__in=[UsedFunctionName.ASSISTANT_MESSAGE_GENERATOR,
                                    UsedFunctionName.USER_MESSAGE_REPHRASER]
        ).values_list("input_tokens", flat=True))
        message_output_token_count = sum(statistics_data.filter(
            used_function_name__in=[UsedFunctionName.ASSISTANT_MESSAGE_GENERATOR,
                                    UsedFunctionName.USER_MESSAGE_REPHRASER]
        ).values_list("completion_tokens", flat=True))
        web_summarizer_input_token_count = sum(statistics_data.filter(
            used_function_name__in=[UsedFunctionName.WEB_PAGE_SUMMARY_CREATION]).values_list(
                "input_tokens", flat=True))
        web_summarizer_output_token_count = sum(statistics_data.filter(
            used_function_name__in=[UsedFunctionName.WEB_PAGE_SUMMARY_CREATION]
        ).values_list("completion_tokens", flat=True))
        image_summarizer_input_token_count = sum(statistics_data.filter(
            used_function_name__in=[UsedFunctionName.IMAGE_SUMMARY_CREATION]).values_list(
                "input_tokens", flat=True))
        image_summarizer_output_token_count = sum(statistics_data.filter(
            used_function_name__in=[UsedFunctionName.IMAGE_SUMMARY_CREATION]
        ).values_list("completion_tokens", flat=True))
        embedding_token_count = sum(TokenUsage.objects.filter(
            used_function_name__in=[UsedFunctionName.EMBEDDING_CREATION]
        ).values_list(
            "total_tokens", flat=True))
        chat_message_count = ChatMessage.objects.filter(
            organization_id=organization_id,
            agent=Agent.USER
        ).count()
        assistant_message_count = ChatMessage.objects.filter(
            organization_id=organization_id,
            agent=Agent.ASSISTANT
        ).count()
        return Response(data={
            "organizationId": organization_id,
            "messageInputFullTokenCount": message_input_token_count,
            "messageOutputFullTokenCount": message_output_token_count,
            "webSummarizerInputFullTokenCount": web_summarizer_input_token_count,
            "webSummarizerOutputFullTokenCount": web_summarizer_output_token_count,
            "imageSummarizerInputFullTokenCount": image_summarizer_input_token_count,
            "imageSummarizerOutputFullTokenCount": image_summarizer_output_token_count,
            "embeddingFullTokenCount": embedding_token_count,
            "chatMessageCount": chat_message_count,
            "assistantMessageCount": assistant_message_count

        }, status=status.HTTP_200_OK)


class ConfigurationView(APIView):
    def put(self, request: Request, *args, **kwargs) -> Response:
        serializer = ConfigurationSerializer(data=request.data)
        if serializer.is_valid():
            validated_data = serializer.validated_data
            configuration = Configuration.objects.get(
                organization=validated_data.get('organization'))
            configuration = serializer.update(
                instance=configuration, validated_data=validated_data)
            data = serializer.to_representation(configuration)
            response = get_resource_updated_response(
                resource="Configuration", data=data)
            logger.info(
                f"{response.message} - [{configuration.organization.id}] - {configuration.id}")
            return response.response

        response = get_invalid_request_response(data=serializer.errors)
        logger.info(f"{response.message} {response.status_code}")
        return response.response

    def get(self, request: Request, *args, **kwargs) -> Response:
        organization_id = request.query_params.get('organization')
        logger.info(f"Fetching configuration - [{organization_id}]")

        serializer = ConfigurationSerializer(
            data=request.query_params)
        if serializer.is_valid():
            organization = serializer.validated_data.get('organization')
            configuration = Configuration.objects.get(
                organization=organization)
            data = serializer.to_representation(configuration)
            logger.info(
                f"configuration fetched - [{organization.id}] - {configuration.id}")
            return Response(data=data, status=status.HTTP_200_OK)

        response = get_invalid_request_response(data=serializer.errors)
        logger.info(f"{response.message} {response.status_code}")
        return response.response


class WebPagesListView(APIView):
    def get(self, request: Request, *args, **kwargs) -> Response:
        organization_id = request.query_params.get('organization')
        if not organization_id:
            raise invalid_request(param='organization')

        web_pages = ProductWebPage.objects.filter(
            organization_id=organization_id)
        chat_history = WebPageSerializer(web_pages, many=True).data
        return Response(data=chat_history, status=status.HTTP_200_OK)

    def post(self, request: Request, *args, **kwargs) -> Response:
        serializer = WebPageSerializer(data=request.data, many=True)

        if not serializer.is_valid():
            response = get_invalid_request_response(
                message=serializer.errors)
            logger.info(response.message)
            return response.response

        serializer.save()
        response = get_resource_created_response(resource='WebPages')
        logger.info(f"Bulk - {response.message}")
        return response.response

    def delete(self, request: Request, *args, **kwargs) -> Response:
        data = request.data

        if self.validate_delete_response(data):
            for i in data:
                ProductWebPage.objects.filter(
                    organization_id=i.get('organization'),
                    url=i.get('url')
                ).delete()
            response = get_resource_deleted_response(resource='WebPages')
            logger.info(f'{response.message}')
            return response.response

    def validate_delete_response(self, data: list) -> bool:
        required_fields = ['organization', 'url']
        for i in data:
            if not all(item in i for item in required_fields):
                raise invalid_request('organization, url')
        return True
