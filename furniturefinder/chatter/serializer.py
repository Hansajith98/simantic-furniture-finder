from rest_framework.serializers import ModelSerializer

from chatter.models import ChatMessage


class ChatHistorySerializer(ModelSerializer):
    def to_representation(self, instance):
        return {
            'id': instance.id,
            'sessionId': instance.session_id,
            'userId': instance.user_id,
            'messageTimestamp': instance.user_id if instance.timestamp else "",
            'role': instance.agent if instance.agent else "",
            'message': instance.message if instance.message else "",
            'ragType': instance.rag_type,
            'is_message': instance.is_image,
        }
