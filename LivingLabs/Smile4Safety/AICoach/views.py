from django.http import JsonResponse, HttpResponse
from django.shortcuts import render

# Create your views here.

import json
from json import dumps
from datetime import datetime, timezone

from .models import ModelSpecification
from pathlib import Path
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import pipeline

combinationFunctions = []

#for english sentiment analysis
model_name = "siebert/sentiment-roberta-large-english"
#use this to create pipeline and use model (takes time)
#sentiment_analysis = pipeline("sentiment-analysis",model= model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name)
tokenizer = RobertaTokenizer.from_pretrained(model_name)

sentiment_analysis = pipeline("sentiment-analysis",model=model, tokenizer = tokenizer)
#Example: print(sentiment_analysis("I love this!"))

#for dutch
model_name = "DTAI-KULeuven/robbert-v2-dutch-sentiment" #"pdelobelle/robbert-v2-dutch-base"
model = RobertaForSequenceClassification.from_pretrained(model_name)
tokenizer = RobertaTokenizer.from_pretrained(model_name)

classifier = pipeline('sentiment-analysis', model=model, tokenizer = tokenizer)

#example
#    result1 = classifier('Gefeliciteerd, Ik vind het mooi')
#    result2 = classifier('Ik vind het lelijk')
#    print(result1)
#    print(result2)

from django.conf import settings
from django.utils import translation



def renderPage(request, page,dictelement):
    user_language = 'en'
    translation.activate(user_language)
    response = render(request, page, dictelement)
    response.set_cookie(settings.LANGUAGE_COOKIE_NAME, user_language)

    return response


def main(request):
    #main function
    #return modelcreation(request)
    #return messagespecification(request)
    #return modelediting(request)

    #return render(request, 'testfile.html', {})
    return monitoring(request)

    #return createmodel(request)
    #return eventbasedmonitoring(request)
    return JsonResponse({"status": 'END OK'})

##### Model specification with reference to world states
##### Model monitoring

def monitoring(request):
    ''' This function is used to monitor the model execution.
        This renders the model with actionable and observation based states.
        The states which are actionable have a message associated at the successful completion of the action.
    '''
    #fetches the last model monitoring_state
    modelobj = ModelSpecification.objects.filter().order_by('-model_id')[0]
    #modelobj = ModelSpecification.objects.get(model_id=15)


    specs = modelobj.model_specification

    jsonData = json.loads(specs)
    dict_model = {
        'id': modelobj.model_id,
        'name': modelobj.model_name,
        'specification': jsonData
    }

    dataJSON = json.dumps(dict_model)  # dict to str
    #return render(request, 'testfile.html', {'data': dataJSON})
    return render(request, 'monitoring.html', {'data': dataJSON})

def modelcreation(request):
    '''This function shows the interface to create the model.
    Algorithm:
        1. Load the combinationFunctions that will be used for fuction specifications for a model. It can be from DB.
        It might use addcombinationfunction interface for this.
        '''

    cwd = Path.cwd()
    filename = str(cwd) + "\\AICoach\\templates\\combinationfunctionlibrary.json"
    with open(filename) as json_file:
        combinationFunctions = json.load(json_file)
        # print(combinationFunctions)
        dataJSON = dumps(combinationFunctions)

    return render(request, 'modelcreation.html', {'data': dataJSON})

def modelspecification(request):
    '''This function writes the model specification to the database.
    It is called from modelcreation
    '''
    # writing new model to the database
    if request.method == 'POST':
        data = request.body  # retrieving model in bytes

        # Decode UTF-8 bytes to Unicode, and convert single quotes
        # to double quotes to make it valid JSON
        str_model = data.decode('utf8').replace("'", '"')  # returns byte data as string
        dict_model = json.loads(str_model)  # dict
        name = dict_model["name"]
        specs = dict_model["stateMatrix"]

        # validate and database save

        jsonspecs = json.dumps(specs)

        ModelSpecification.objects.create(model_name=name, model_specification=jsonspecs,
                                          last_modified=datetime.now(timezone.utc))
        return JsonResponse({"status": 'ModelSpecification object created - success messagespecification'})

    else:
        return JsonResponse({"status": 'no model received'})

def messagespecification(request):
    '''This function shows the interface to create the messages samples for all the worldly states.
        1. actionable/observatory
        2. messages regarding the states
        Replica of actionspecification

        Save URL of messagesspecification
        '''
    modelobj = ModelSpecification.objects.filter().order_by('-model_id')[0]

    specs = modelobj.model_specification

    ##formatting string to make it json

    jsonData = json.loads(specs)

    dict_model = {
        'id': modelobj.model_id,
        'name': modelobj.model_name,
        'specification': jsonData
    }

    dataJSON = json.dumps(dict_model)  # dict to str

    return render(request, 'messagespecification.html', {'data': dataJSON})

def modelediting(request):
    #This method shows the complete model
    cwd = Path.cwd()
    filename = str(cwd) + "\\AICoach\\templates\\combinationfunctionlibrary.json"
    combinationFunctions = []
    with open(filename) as json_file:
        combinationFunctions = json.load(json_file)

    #loading the model
    modelobj = ModelSpecification.objects.filter().order_by('-model_id')[0]
    #modelobj = ModelSpecification.objects.get(model_id=10)

    specs = modelobj.model_specification

    jsonData = json.loads(specs)
    dict_model = {
        'id': modelobj.model_id,
        'name': modelobj.model_name,
        'specification': jsonData,
        'combinationFunctions':combinationFunctions
    }

    dataJSON = json.dumps(dict_model)  # dict to str

    return render(request, 'modelediting.html', {'data': dataJSON})

def specificationedited(request):
    # This method edits the complete model
    if request.method == 'POST':
        data = request.body  # retrieving model in bytes

        # Decode UTF-8 bytes to Unicode, and convert single quotes
        # to double quotes to make it valid JSON
        str_model = data.decode('utf8').replace("'", '"')  # returns byte data as string
        dict_model = json.loads(str_model)  # dict

        id = dict_model["id"]
        name = dict_model["name"]
        modelobj = ModelSpecification.objects.get(model_id=id)

        specs = dict_model["stateMatrix"]

        modelobj.model_specification = json.dumps(specs)
        modelobj.save()

        # validate and database save

        return JsonResponse({"status": 'success to editting'})

def modelspecificationupdated(request):
    #TODO: CHECK it!!!This method is called when the model specification has been updated
    if request.method == 'POST':
        data = request.body  # retrieving model in bytes

        # Decode UTF-8 bytes to Unicode, and convert single quotes
        # to double quotes to make it valid JSON
        str_model = data.decode('utf8').replace("'", '"')  # returns byte data as string
        dict_model = json.loads(str_model)  # dict

        id = dict_model["id"]
        name = dict_model["name"]
        modelobj = ModelSpecification.objects.get(model_id=id)

        specs = dict_model["stateMatrix"]

        modelobj.model_specification = json.dumps(specs)
        modelobj.save()

        return JsonResponse({"status": 'success to update'})


def messagespecificationupdated(request):
    '''This function is replica of updatespecification algorithm is
        1. updates the role of each of the states
        2. analyze the sentiments regarding the state name and the messages in it.
        Therefore, it not only updates the roles of each state, but also updates the sentiments for the dialog'''

    if request.method == 'POST':
        data = request.body  # retrieving model in bytes

        # Decode UTF-8 bytes to Unicode, and convert single quotes
        # to double quotes to make it valid JSON
        str_model = data.decode('utf8').replace("'", '"')  # returns byte data as string
        dict_model = json.loads(str_model)  # dict

        id = dict_model["id"]
        name = dict_model["name"]
        modelobj = ModelSpecification.objects.get(model_id=id)

        specs = dict_model["stateMatrix"]

        # analyze the specs and set the sentiments
        specs = analyzesentiments(specs)
        modelobj.model_specification = json.dumps(specs)
        modelobj.save()

        return JsonResponse({"status": 'success to update'})

def supportspecificationupdated(request):
    '''url of messagespecification'''
    return JsonResponse({"status": 'Roles updated . Show success interface'})



#Sentiment Analysis
def getsentiment(string):
    ''' This function returns the maximum sentiment among positive, negative and neutral.'''
    results = classifier(string)
    maxelem = {"label": '', "score": 0}
    for result in results:
        score = result['score']
        if score > maxelem['score']:
            maxelem = result
    return maxelem

def analyzesentiments(list_model):
    '''This function analyzes the sentiment of the states of the model'''
    #str_model = modelobj.model_specification

    #dict_model = json.loads(str_model)  # dict
    base_model = list_model[0]

    states = base_model['states']
    for state in states:
        id = state['id']
        name = state['name']
        '''sentiment analysis 
            1. if state name is negative then set sentiment.
            2. else if sucess message is negative then set sentiment as negative.
        '''
        sentiment = getsentiment(name)

        if sentiment['label'] == 'Positive':
            message = state['message']
            sentiment = getsentiment(message)

        state['sentiment_lbl'] = sentiment['label']

    base_model['states'] = states
    list_model[0] = base_model

    return list_model #returns list


#Simulation related functions

def getstatestatus(request):
    #REF: https://stackoverflow.com/questions/43708387/django-display-json-or-httpresponse-in-template
    data = request.session['progress']
    #print('Views: getstatestatus === sent data')
    #print(type(data))
    #print(data)


    return JsonResponse(data,safe = False)
    #return HttpResponse(status=204)


#global progressOfNetwork

from .simulation_controller import *

def updateModelSpecification(model_input,state):
    for input in model_input:
        stateinfo = input['state']
        id = stateinfo['id']
        if(state.getid() == id):
            stateinfo['observed'] = state.observed
            outputinfo = state.getLastOutput()
            cur_val = outputinfo.getValue()
            time_stamp = outputinfo.getTimeStamp()
            stateinfo['values'].append({'cur_value':cur_val , 'time_stamp': time_stamp})
    x = 10

def updateProgressElement(progress_input,state):
    for iteration in progress_input:

        prginfo = iteration['progress']

        for prg in prginfo:
            #Updating from elements
            id = prg['from']['id']
            if (state.getid() == id):
                #if(id == 'X6'):
                #    print("====STATE===")
                #print('state info', state.getid(),state.observed,state.incomingconnections)

                prg['from']['observed'] = state.observed
                outputinfo = state.getLastOutput()
                cur_val = outputinfo.getValue()
                time_stamp = outputinfo.getTimeStamp()
                prg['from']['values'].append({'curvalue':cur_val , 'time_stamp':time_stamp})
                #print('from state',prg['from'])
    x = 10

def setcurrentstatus(request):
    # This function is used to set the status of the states

    if request.method == 'POST':
        data = request.body  # retrieving model in bytes
        #print('data', data)

        statematrix = simulation_controller(data)

        model_input_sample = json.loads(data)  # returns dictionary
        model_input = json.loads(data)  # returns dictionary

        # print("=======Views:Model Input =========================")
        # print(model_input)

        # TODO: simulation results have statematrix
        # convert them into jsonData = json.dumps(simulation_results)
        x = 10
        for state in statematrix:
            level = state.getLevel()
            if state.id == 45:
                x = 10
            if level == 0:
                updateModelSpecification(model_input['base_level'], state)
                updateProgressElement(model_input['last_progress'], state)
            else:
                if level == 1:
                    updateModelSpecification(model_input['first_level'], state)
                else:
                    if level == 2:
                        updateModelSpecification(model_input['second_level'], state)

        # model_output = model_input_sample
        model_output = model_input
        # print("=======Model Output =========================")
        # print(model_output)

        jsonData = json.dumps(model_output)  # returns string

        request.session['progress'] = jsonData
        # print(jsonData)

    return HttpResponse(status=204)

def setstatestatus(request):
    # This function is used to set the status of the states

    if request.method == 'POST':
        data = request.body  # retrieving model in bytes
        statematrix = simulation_controller(data)

        model_input_sample = json.loads(data)  # returns dictionary
        model_input = json.loads(data)  # returns dictionary

        #print("=======Views:Model Input =========================")
        #print(model_input)

        #simulation results have statematrix
        # convert them into jsonData = json.dumps(simulation_results)
        x = 10
        for state in statematrix:
            level = state.getLevel()
            if state.id == 45:
                x = 10
            if level == 0:
                updateModelSpecification(model_input['base_level'],state)
                updateProgressElement(model_input['last_progress'],state)
            else:
                if level == 1:
                    updateModelSpecification(model_input['first_level'],state)
                else:
                    if level == 2:
                        updateModelSpecification(model_input['second_level'],state)

        #model_output = model_input_sample
        model_output = model_input
        #print("=======Model Output =========================")
        #print(model_output)

        jsonData = json.dumps(model_output)  # returns string

        request.session['progress'] = jsonData
        #print(jsonData)



    return HttpResponse(status=204)