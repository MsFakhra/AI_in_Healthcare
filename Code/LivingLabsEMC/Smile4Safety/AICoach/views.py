from django.http import JsonResponse
from django.shortcuts import render
import json
from json import dumps
from datetime import datetime, timezone
from .models import ModelSpecification

# Create your views here.
def main(request):
    #TO-DO: save the file to library, populate and remove open code, rather load from DB and then dump into json file
    filename= 'C:\\Fakhra\\Post-Doc\\Working on Code\\InterfaceDesign\\LivingLabsEMC\\Smile4Safety\\AICoach\\templates\\combinationfunctionlibrary.json'
    with open(filename) as json_file:
        combinationFunctions = json.load(json_file)
        #print(combinationFunctions)
    dataJSON = dumps(combinationFunctions)
    #return render(request, 'monitoring.html', {'data': dataJSON})

    return render(request, 'createmodel.html', {'data': dataJSON})
    #return render(request, 'actionspecification.html', {'data': dataJSON})
    #return render(request, 'addcombinationfunction.html', {'data': dataJSON})
    #return render(request, 'testfile.html', {'data': dataJSON})
    #return render(request,'main.html',{})

def statespecification(request):
    # writing model to the database
    if request.method == 'POST':
        data = request.body     #retrieving model in bytes

        # Decode UTF-8 bytes to Unicode, and convert single quotes
        # to double quotes to make it valid JSON
        str_model = data.decode('utf8').replace("'", '"')    # returns byte data as string
        dict_model = json.loads(str_model) #dict
        str_model = json.dumps(dict_model) #dict to str

        name = dict_model["name"]
        specs = dict_model["stateMatrix"]

        # validate and database save

        ModelSpecification.objects.create(model_name = name, model_specification= specs, last_modified =  datetime.now(timezone.utc))
        return JsonResponse({"status": 'success statespecification'})

    else:
        return JsonResponse({"status": 'no model received'})

def updatespecification(request):
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
        model.model_specification = specs
        model.save()

        return JsonResponse({"status": 'success to update'})

def rolesspecificationupdated(request):
    return JsonResponse({"status": 'success roles'})
def actionspecification(request):
    # retrieve last element from database
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


def adminpanel(request):
    return render(request, 'adminpanel.html', {})

def createmodel(request):
    return render(request, 'createmodel.html', {})

def searchmodel(request):
    return render(request, 'searchmodel.html', {})

