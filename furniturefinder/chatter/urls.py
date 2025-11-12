from django.urls import path
from .views import ChatterView

app_name = 'chatter'

urlpatterns = [
    path('chat/', ChatterView.as_view(), name='chat'),
    path('', ChatterView.as_view(), name='index'),
]
