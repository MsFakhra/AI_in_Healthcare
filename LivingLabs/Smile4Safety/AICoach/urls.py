from django.urls import path
from AICoach import views
from django.urls import path


urlpatterns = [

    # creating the navigation pages
    path('', views.main, name='main'),

    #path('adminpanel/', views.adminpanel, name='adminpanel'),  # admin panel
    #path('searchmodel/', views.searchmodel, name='searchmodel'),

    # designing the model
    path('createmodel/', views.createmodel, name='createmodel'),
    path('statespecification/', views.statespecification, name="statespecification"),
    path('actionspecification/', views.actionspecification, name="actionspecification"),
    #assigning the roles
    path('updatespecification/', views.updatespecification, name="updatespecification"),
    path('rolesspecificationupdated/', views.rolesspecificationupdated, name="rolesspecificationupdated"),


    # monitoring the progress of model
    path('eventbasedmonitoring/', views.eventbasedmonitoring, name="eventbasedmonitoring"),


    ##updated views based on the actionable states
    # designing the model Replica of line 16-line 20
    path('modelcreation/', views.modelcreation, name='modelcreation'),
    path('modelspecification/', views.modelspecification, name="modelspecification"),
    path('messagespecification/', views.messagespecification, name="messagespecification"),
    #assigning the roles
    path('messagespecificationupdated/', views.messagespecificationupdated, name="messagespecificationupdated"),
    path('supportspecificationupdated/', views.supportspecificationupdated, name="supportspecificationupdated"),

    # monitoring the progress of model
    path('monitoring/', views.monitoring, name="monitoring"),





]