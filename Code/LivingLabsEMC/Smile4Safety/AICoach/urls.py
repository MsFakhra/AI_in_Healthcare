from django.urls import path
from AICoach import views

urlpatterns = [
    #creating the navigation pages
    path('', views.main, name='main'),
    path('adminpanel/', views.adminpanel, name='adminpanel'), #admin panel
    path('createmodel/', views.createmodel, name='createmodel'),
    path('searchmodel/', views.searchmodel, name='searchmodel'),

    #designing the model
    path('statespecification/', views.statespecification, name="statespecification"),
    path('actionspecification/', views.actionspecification, name="actionspecification"),

    path('updatespecification/', views.updatespecification, name="updatespecification"),
    path('rolesspecificationupdated/', views.rolesspecificationupdated, name="rolesspecificationupdated"),

    #monitoring the progress of model


]