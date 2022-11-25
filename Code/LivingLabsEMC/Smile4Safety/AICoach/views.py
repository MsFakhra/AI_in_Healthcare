from django.http import JsonResponse
from django.shortcuts import render
import json
from json import dumps
from datetime import datetime, timezone
from .models import ModelSpecification

# Create your views here.
def main(request):
    modelobj = ModelSpecification.objects.filter().order_by('-model_id')[0]

    #modelobj = ModelSpecification.objects.get(model_id=33) #use this model for monitoringv1.html
    #modelobj = ModelSpecification.objects.get(model_id=34)  # use this model for recursive and monitoringv1.html

    specs = modelobj.model_specification

    ##formatting string to make it json
    withoutspaces = specs.replace(" ", "")  # removes spaces
    rightcommas = withoutspaces.replace("'", '"')  # right inverted commas
    withoutTrue = rightcommas.replace("True", '"True"')  # True to 'True'
    withoutFalse = withoutTrue.replace("False", '"False"')  # False to 'False'

    jsonData = json.loads(withoutFalse)

    dict_model = {
        'id': modelobj.model_id,
        'name': modelobj.model_name,
        'specification': jsonData
    }

    dataJSON = json.dumps(dict_model)  # dict to str

    return render(request, 'monitoring.html', {'data': dataJSON})
    #return render(request, 'monitoringv1.html', {'data': dataJSON})

    #checked
    #return render(request, 'createmodel.html', {'data': dataJSON})
    #return render(request,'main.html',{})


#monitoring

def monitoring(request):

    return render(request, 'monitoring.html', '')



#desinging model
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
    #updating the roles
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
        model.model_specification = json.dumps(specs)
        model.save()

        return JsonResponse({"status": 'success to update'})

def rolesspecificationupdated(request):
    return JsonResponse({"status": 'Roles updated . Show success interface'})
def actionspecification(request):
    # retrieve last element from database. The aim si to set the action specification of the model
    modelobj = ModelSpecification.objects.filter().order_by('-model_id')[0]

    specs = modelobj.model_specification

    ##formatting string to make it json
    withoutspaces = specs.replace(" ", "")  #removes spaces
    rightcommas = withoutspaces.replace("'", '"')   #right inverted commas
    withoutTrue = rightcommas.replace("True", '"True"') #True to 'True'
    withoutFalse = withoutTrue.replace("False", '"False"')  #False to 'False'

    jsonData = json.loads(withoutFalse)

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
    filename = 'C:\\Fakhra\\Post-Doc\\Working on Code\\InterfaceDesign\\LivingLabsEMC\\Smile4Safety\\AICoach\\templates\\combinationfunctionlibrary.json'
    with open(filename) as json_file:
        combinationFunctions = json.load(json_file)
        # print(combinationFunctions)
    dataJSON = dumps(combinationFunctions)

    return render(request, 'createmodel.html', {'data': dataJSON})

def searchmodel(request):
    return render(request, 'searchmodel.html', {})

