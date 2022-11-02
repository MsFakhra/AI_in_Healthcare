from django.urls import path
from AICoach import views

urlpatterns = [
    path('', views.main, name='main'),
    path('adminpanel/', views.adminpanel, name='adminpanel'), #admin panel
    path('createmodel/', views.createmodel, name='createmodel'),
    path('searchmodel/', views.searchmodel, name='searchmodel'),

    path('statespecification/', views.statespecification, name="statespecification"),
    path('actionspecification/', views.actionspecification, name="actionspecification"),

    path('updatespecification/', views.updatespecification, name="updatespecification"),
    path('rolesspecificationupdated/', views.rolesspecificationupdated, name="rolesspecificationupdated"),



]