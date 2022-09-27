from django.urls import path
from AICoach import views

urlpatterns = [
    path('', views.main, name='main'),
]