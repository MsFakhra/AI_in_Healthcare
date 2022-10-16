from django.http import JsonResponse
from django.shortcuts import render
import json
from json import dumps

# Create your views here.
def main(request):
    #TO-DO: save the file to library, populate and remove open code, rather load from DB and then dump into json file
    filename= 'C:\\Fakhra\\Post-Doc\\Working on Code\\InterfaceDesign\\LivingLabsEMC\\Smile4Safety\\AICoach\\templates\\combinationfunctionlibrary.json'
    with open(filename) as json_file:
        combinationFunctions = json.load(json_file)
        #print(combinationFunctions)
    dataJSON = dumps(combinationFunctions)
    return render(request, 'createmodel.html', {'data': dataJSON})
    #return render(request, 'addcombinationfunction.html', {'data': dataJSON})
    #return render(request, 'testfile.html', {'data': dataJSON})
    #return render(request,'main.html',{})

def statespecification(request):
    if request.method == 'POST':
        data = request.body     #retrieving model
        # Decode UTF-8 bytes to Unicode, and convert single quotes
        # to double quotes to make it valid JSON
        my_json = data.decode('utf8').replace("'", '"')
        dataJSON = dumps(my_json)
        print(dataJSON)

        # validate and database save
        return JsonResponse({"status": 'success'})

    else:
        return JsonResponse({"status": 'no model received'})
    #status = model_generation(data)
    #return JsonResponse({"status": status})
def actionspecification(request):
    # retrieve database
    return render(request, 'actionspecification.html', {})


def adminpanel(request):
    return render(request, 'adminpanel.html', {})

def createmodel(request):
    return render(request, 'createmodel.html', {})

def searchmodel(request):
    return render(request, 'searchmodel.html', {})

