from django.http import JsonResponse
from django.shortcuts import render
import json
from json import dumps
from datetime import datetime, timezone
from .models import ModelSpecification

# Create your views here.
def main(request):

    modelobj = ModelSpecification.objects.filter().order_by('-model_id')[0]

    #modelobj = ModelSpecification.objects.get(model_id=36)  # ss specification
    #modelobj = ModelSpecification.objects.get(model_id=40)  # works with eventbasedmonitoring recursive - english model specification
    #modelobj = ModelSpecification.objects.get(model_id=39)  # use this model it has x1,x2 specification without recursive works with monitor

    #modelobj = ModelSpecification.objects.get(model_id=34)  # use this model it has x1,x2 specification without recursive
    #modelobj = ModelSpecification.objects.get(model_id=32)  # use this model for recursive and monitoringv1.html it has x1,x2 specification

    specs = modelobj.model_specification
    '''
    ##formatting string to make it json
    withoutspaces = specs.replace(" ", "")  # removes spaces
    rightcommas = withoutspaces.replace("'", '"')  # right inverted commas
    withoutTrue = rightcommas.replace("True", '"True"')  # True to 'True'
    withoutFalse = withoutTrue.replace("False", '"False"')  # False to 'False'

    jsonData = json.loads(withoutFalse)
    '''

    jsonData = json.loads(specs)
    dict_model = {
        'id': modelobj.model_id,
        'name': modelobj.model_name,
        'specification': jsonData
    }

    dataJSON = json.dumps(dict_model)  # dict to str

    return render(request, 'actionspecification.html', {'data': dataJSON})

    #return render(request, 'testfile.html', {'data': dataJSON})  trying to get audio

    return render(request, 'eventbasedmonitoring.html', {'data': dataJSON})  #100% working with one direction tested
    #return render(request, 'monitoring.html', {'data': dataJSON})  #100% working with one direction tested

    #return render(request, 'monitoringv2.html', {'data': dataJSON})  #new working version with recordfrom and recordto
    #return render(request, 'monitoringv1.html', {'data': dataJSON})  #oldest working version

    #checked
    #return render(request, 'createmodel.html', {'data': dataJSON})
    #return render(request,'main.html',{})


#monitoring

def monitoring(request):
    ''' This function is used to monitor the model execution.
        This renders the model with actionable and observation based states.
        The states which are actionable have a message associated at the successful completion of the action.
    '''
    #fetches the last model
    modelobj = ModelSpecification.objects.filter().order_by('-model_id')[0]

    specs = modelobj.model_specification

    jsonData = json.loads(specs)
    dict_model = {
        'id': modelobj.model_id,
        'name': modelobj.model_name,
        'specification': jsonData
    }

    dataJSON = json.dumps(dict_model)  # dict to str

    return render(request, 'eventbasedmonitoring.html', {'data': dataJSON})  # 100% working with one direction tested


#designing model
def addcombinationfunction(request):
    # TODO: save the file to library, populate and remove open code, rather load from DB and then dump into json file
    # still have to create interface + DB to add combination functions
    return JsonResponse({"status": 'still have to create interface + DB to add combination functions'})
    # return render(request, 'addcombinationfunction.html', {'data': dataJSON})

def statespecification(request):
    # writing model to the database
    if request.method == 'POST':
        data = request.body     #retrieving model in bytes

        # Decode UTF-8 bytes to Unicode, and convert single quotes
        # to double quotes to make it valid JSON
        str_model = data.decode('utf8').replace("'", '"')    # returns byte data as string
        dict_model = json.loads(str_model) #dict
        name = dict_model["name"]
        specs = dict_model["stateMatrix"]

        # validate and database save

        jsonspecs = json.dumps(specs)

        ModelSpecification.objects.create(model_name = name, model_specification= jsonspecs, last_modified =  datetime.now(timezone.utc))
        return JsonResponse({"status": 'success statespecification'})

    else:
        return JsonResponse({"status": 'no model received'})

def updatespecification(request):
    '''This function analyze the sentiments regarding the state name and the messages in it.
    Therefore, it not only updates the roles of each state, but also updates the sentiments for the dialog'''

    if request.method == 'POST':
        data = request.body     #retrieving model in bytes


        # Decode UTF-8 bytes to Unicode, and convert single quotes
        # to double quotes to make it valid JSON
        str_model = data.decode('utf8').replace("'", '"')    # returns byte data as string
        dict_model = json.loads(str_model) #dict
        str_model = json.dumps(dict_model) #dict to str

        id = dict_model["id"]
        name = dict_model["name"]
        model = ModelSpecification.objects.get(model_id=id)

        specs = dict_model["stateMatrix"]

        #analyze the specs and set the sentiment
        analyzesentiments(specs)


        #this updates the model specification with sentiment
        #model.model_specification = json.dumps(specs)
        #model.save()

        return JsonResponse({"status": 'success to update'})

def rolesspecificationupdated(request):
    return JsonResponse({"status": 'Roles updated . Show success interface'})
def actionspecification(request):
    # retrieve last element from database. The aim is to set the action specification of the model
    modelobj = ModelSpecification.objects.filter().order_by('-model_id')[0]

    specs = modelobj.model_specification

    ##formatting string to make it json
    '''withoutspaces = specs.replace(" ", "")  #removes spaces
    rightcommas = withoutspaces.replace("'", '"')   #right inverted commas
    withoutTrue = rightcommas.replace("True", '"True"') #True to 'True'
    withoutFalse = withoutTrue.replace("False", '"False"')  #False to 'False'
    
    jsonData = json.loads(withoutFalse)
    '''
    jsonData = json.loads(specs)

    dict_model = {
        'id': modelobj.model_id,
        'name': modelobj.model_name,
        'specification': jsonData
    }

    dataJSON = json.dumps(dict_model) #dict to str

    return render(request, 'actionspecification.html', {'data': dataJSON})

#panel
def adminpanel(request):
    return render(request, 'adminpanel.html', {})

def createmodel(request):
    '''This function shows the interface to create the model.
    Algorithm:
        1. Load the combinationFunctions that will be used for fuction specifications for a model. It can be from DB.
        It might use addcombinationfunction interface for this.
    '''
    filename = 'C:\\Fakhra\\Post-Doc\\Working on Code\\InterfaceDesign\\LivingLabsEMC\\Smile4Safety\\AICoach\\templates\\combinationfunctionlibrary.json'
    with open(filename) as json_file:
        combinationFunctions = json.load(json_file)
        # print(combinationFunctions)
    dataJSON = dumps(combinationFunctions)

    return render(request, 'createmodel.html', {'data': dataJSON})

def searchmodel(request):
    return render(request, 'searchmodel.html', {})

#supporting methods
def analyzesentiments(specs):
    '''This function analyzes the following to generate the sentiment of the speech.
        1. state names
        2. state success and warning messages.
    Algorithm:
        1. Detect the language of the model
        2. Perform the respective Sentiment analysis
        3. If sentiment is negative for certain state. Then regulation is required.

    '''
    specification = specs

    #extract states from the base layer

    baselayer = specs[0]
    firstlayer = specs[1]
    secondlayer = specs[2]

    states = baselayer["states"]

    for state in states:
        name = state["name"]
        successmsg = state["successmsg"]
        language =


    print(specs)
