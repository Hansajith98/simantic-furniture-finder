from django.urls import path
from . import views

app_name = 'furnituredescriptor'

urlpatterns = [
    path('', views.index, name='index'),
]