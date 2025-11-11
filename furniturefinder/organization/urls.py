from django.urls import path, re_path
from .views import *
urlpatterns = [
    path('ping/', TestView.as_view(), name="test"),
    path('prompt/', PromptView.as_view(), name='prompt'),
    path('statistics/', OrganizationStatInfoView.as_view(), name='statistics'),
    path('configuration/', ConfigurationView.as_view(), name='configuration'),
    path('general-info/', GeneralInfoView.as_view(), name='general-info'),
    path('webpage/', WebPagesListView.as_view(), name='webpage'),
    path('<str:id>/', OrganizationView.as_view(), name='organization'),
    path('', OrganizationView.as_view(), name='organization'),
]
