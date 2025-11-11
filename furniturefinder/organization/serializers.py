import logging
import uuid

from rest_framework import serializers
from organization.models import ProductWebPage, ProductImage, Organization, \
    GeneralInfo, ProcessState, Configuration, ProductCollection, MappingProductCollection
from utils.response import *


class WebPageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProductWebPage
        fields = '__all__'


class ProductImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProductImage
        fields = '__all__'


class GeneralInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = GeneralInfo
        fields = ['general_information', 'vectorization_status', 'updated_on',
                  'created_on']

    def validate(self, data):
        """
        Custom validation to check if the Organization exists and has a linked 
        GeneralInfo.
        """
        organization_id = self.context.get("organization_id")
        if not organization_id:
            raise serializers.ValidationError(
                {"organization_id": "Organization ID is required."})

        try:
            organization = Organization.objects.get(id=organization_id)
        except Organization.DoesNotExist:
            raise serializers.ValidationError(
                {"organization_id": "Organization not found."})

        if organization.general_info:
            self.instance = organization.general_info

        self.context["organization"] = organization

        return super().validate(data)

    def create(self, validated_data) -> CustomResponse:
        """
        Create a new GeneralInfo instance if necessary and link it to the 
        Organization.
        """
        organization = self.context.get("organization")

        if organization.general_info:
            return get_resource_exists_response(resource="general info")

        (general_info, created) = GeneralInfo.objects.get_or_create(
            **validated_data)
        organization.general_info = general_info
        organization.save()

        return general_info

    def update(self, instance, validated_data) -> CustomResponse:
        """
        Update the GeneralInfo instance with the new general_information data.
        """
        instance.general_information = validated_data.get(
            'general_information', instance.general_information)
        instance.vectorization_status = ProcessState.WAITING
        instance.save()

        return instance


class ConfigurationSerializer(serializers.ModelSerializer):

    class Meta:
        model = Configuration
        fields = ['organization', 'sitemap_configuration', 'scrape_configuration',
                  'image_summary_required', 'knowledge_base_type',
                  'additional_configurations']

        def validate(self, data):
            """
            Custom validation to check if the Organization exists and has a linked 
            GeneralInfo.
            """
            organization_id = data.get("organization")
            if not organization_id:
                raise serializers.ValidationError(
                    {"organization_id": "Organization ID is required."})

            organization = Organization.objects.get(id=organization_id)
            data['organization'] = organization

            return super().validate(data)


class ChatHistorySerializer(serializers.ModelSerializer):
    def to_representation(self, instance):
        return {
            'id': instance.id,
            'organizationId': instance.organization.id,
            'organizationName': instance.organization.name or "",
            'sessionId': instance.session_id,
            'userId': instance.user_id,
            'messageTimestamp': instance.timestamp or "",
            'role': instance.agent or "",
            'message': instance.message or "",
            'ragType': instance.rag_type,
            'is_message': instance.is_image,
        }


class WebPageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProductWebPage
        fields = ['organization', 'url']

    def delete(self, **kwargs):
        assert hasattr(self, '_errors'), (
            'You must call `.is_valid()` before calling `.save()`.'
        )

        assert not self.errors, (
            'You cannot call `.save()` on a serializer with invalid data.'
        )

        assert not hasattr(self, '_data'), (
            "You cannot call `.save()` after accessing `serializer.data`."
            "If you need to access data before committing to the database then "
            "inspect 'serializer.validated_data' instead. "
        )

        validated_data = {**self.validated_data, **kwargs}
        logger = logging.getLogger('chat-with-website')
        logger.info(f'fojasfas - {validated_data}')

        ProductWebPage.objects.filter(
            Organization=validated_data.get('organization'),
            url=validated_data.get('url')
        ).delete()

    def create(self, validated_data):
        """
        Bulk create or update WebPages
        """
        data = validated_data
        organization = data.get('organization')
        instance, created = ProductWebPage.objects.get_or_create(
            id=str(uuid.uuid4()),
            url=data['url'],
            organization=data['organization']
        )
        if not created:
            instance.scrape_status = ProcessState.WAITING
            instance.save()
        else:
            if (url := data.get('collection')):
                try:
                    collection = ProductCollection.objects.get(
                        url=url,
                        organization=organization
                    )
                except ProductCollection.DoesNotExist:
                    raise serializers.ValidationError(
                        f'Collection does not exist - {url}')
            else:
                collection = ProductCollection.objects.get(
                    organization=organization,
                    url='default//:default-collection'
                )
            MappingProductCollection.objects.create(
                product=instance,
                collection=collection
            )

    def to_representation(self, instance):
        return {
            'organizationId': instance.organization.id,
            'url': instance.url,
            'pageTitle': instance.page_title or '',
            'lastScraped': instance.scraped_on
        }
