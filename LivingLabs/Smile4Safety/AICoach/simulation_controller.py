import re, os
import numpy as np
import math
import json as json
from datetime import datetime
################# Controller methods

def simulation_controller(data):
    print('generating simulation')
    statematrix = generateStateInformation(data)
    generateSimulationResults(statematrix)
    return statematrix


#implementation of simulation software

def generateStateInformation(data):
    #This function generates the state information

    input = json.loads(data)  # returns dictionary

    statematrix = []
    level = -1
    for key, value in input.items():
        #print(key, ":", value)
        if (key == 'base_level'):
            model_input = value #returning dictionary having states
            level = 0
        else:
            if (key == 'first_level'):
                model_input = value#['states'] #returning list
                level = 1
            else:
                if (key == 'second_level'):
                    model_input = value#['states']
                    level = 2


        if (key == 'base_level' or key == 'first_level' or key == 'second_level'):
            if(model_input):
                for info in model_input:
                    element = info
                    state = element['state']
                    index = state['id']
                    id = state['id']  # X1
                    if id == 'X20':
                        x = 10

                    name = state['name']
                    in_connections = state['in_connection']  # return list of dictionaries
                    connection_weights = state['connection_weights']

                    if not (is_valid_float(state['speed'])):
                        speed = state['speed']
                    else:
                        speed = float(state['speed'])


                    init_value = float(state['init_value'])
                    last_value = state['values'][-1]
                    combination_functions = state['combination_functions']
                    is_internal = state['is_internal']

                    index = extractId(id)  # returns id as 1
                    incomingconnections = extractIncomingConnections(state,in_connections)  # return incomingconnections
                    connectionweight = extractIncomingConnectionWeights(state,connection_weights, in_connections)  # returns connection weights
                    cfv_structure = extractCFVStructure(combination_functions)

                    # if state is from other levels make them complete
                    if key == 'first_level' or key == 'second_level':
                        complete = True
                    else:
                        complete = state['complete']
                    #initial_state = state['initial_state']
                    state = State(id, index, name, incomingconnections, connectionweight, speed, init_value, cfv_structure,level,complete,is_internal)#,initial_state)
                    statematrix.append(state)


                    # set the initial values from the previous simulation results
                    time_stamp = last_value['time_stamp']
                    last_output = last_value['cur_value']
                    if(np.isnan(last_output)):
                        last_output = 0
                    soutput = StateOutput(time_stamp, last_output)
                    state.setOutputValues(soutput)

    return statematrix

####SimulationResults####
def generateSimulationResults(statematrix):
    print('This generates the simulation for single iteration')

    curriteration = 1
    last_iteration = 0
    cur_time_stamp = datetime.now()
    for state in statematrix:  # j for all states in the state matrix
        index = state.index
        #print('state index,name',index,state.name,state.complete)
        if index == 23:
            x = 10
        speed = state.speed
        connection_weights = state.connection_weights
        cfv_structure = state.cfv_structure
        stateinput = []
        incomingconnections = state.incomingconnections

        completionstatus = state.complete
        inc_completed = []
        # updating the state connections with values the 'b' matrix
        for connection in incomingconnections:
            #print('connection',connection)
            if not (is_valid_float(connection)):
                statevalue = getStateOutputValues(connection, statematrix)#get the last value
                inc_completionstatus = getIncomingCompletionStatus(connection,statematrix,state)
                inc_completed.append(inc_completionstatus)
                #print('completion status ', ind_completionstatus)
                #if(ind_completionstatus == True)
                #if((inc_completionstatus == True and completionstatus == False) or (ind_completionstatus == False)):

                stateinput.append(statevalue)
            else:
                stateinput.append(0)
        stateinput = np.array(stateinput)
        status = getCompletionStatus(inc_completed)
        if (status != 'Unknown'):
            completionstatus = status


        # updating the speed s matrix
        """ if not(isnan(msa(j, 1)))
            s(j, k) = X(msa(j, 1), k);   #k is the value at certain iteration"""
        if not (is_valid_float(speed)):
           adaptivestate = int(findAdaptiveState(speed))
           adaptivestate_speed = statematrix[adaptivestate - 1]
           speed = adaptivestate_speed.getOutputValues()#curriteration)  # state.speed

        # updating the connection weights cw matrix
        for weightindex in range(len(connection_weights)):
            weight = connection_weights[weightindex]
            if not (is_valid_float(weight)):
                adaptivestate = int(findAdaptiveState(weight))
                adaptivestate_weight = statematrix[adaptivestate - 1].getOutputValues()#curriteration)
                connection_weights[weightindex] = adaptivestate_weight
        connection_weights = np.array(connection_weights)
        # state.setConnectionWeight(connection_weights)

        # updating the connection functions and parameters  mcfw and mcfp matrices
        updated_cfv_structure = []
        for sindex in range(len(cfv_structure)):
            weight = cfv_structure[sindex].impactweight
            if weight != 0:  # if certain fucntion is chosen, then look for parameters otherwise ignore it
                id = cfv_structure[sindex].id
                name = cfv_structure[sindex].name
                noOfParams = cfv_structure[sindex].numberOfParams
                params = cfv_structure[sindex].params
                updated_cfv_element = CombiationFunctionStructure(id, name, noOfParams, weight)
                if not (is_valid_float(weight)):
                    adaptivestate = int(findAdaptiveState(weight))
                    adaptivestate_pweight = statematrix[adaptivestate - 1].getOutputValues()#curriteration)
                    updated_cfv_element.setWeight(adaptivestate_pweight)
                else:
                    updated_cfv_element.setWeight(float(weight))

                # if any of the params are adaptive they are updated
                aparams = []
                for param in params:
                   if not (is_valid_float(param)):
                        adaptivestate = int(findAdaptiveState(param))
                        adaptivestate_param = statematrix[adaptivestate - 1].getOutputValues()#curriteration)
                        aparams.append(adaptivestate_param)
                   else:
                        aparams.append(float(param))
                aparams = np.array(aparams)
                updated_cfv_element.setParams(aparams)
                updated_cfv_structure.append(updated_cfv_element)
            # state.setCombinationFunctionStructure(updated_cfv_structure)

        # Generating the output
        """for m=1:1:nocf
               cfv(j,m,k) = bcf(mcf(m), squeeze(cfp(j, :, m, k)), squeeze(cw(j, :, k)).*squeeze(b(j, :, k)));
            end"""
        # print(index, connection_weights, len(connection_weights))
        # print(index, stateinput,len(stateinput))
        singleimpactmatrix = stateinput * connection_weights  # singleimpactmatrix = b * cw = X * Omega = stateinput * connection_weights => it should be int
        cfv_structure = updated_cfv_structure  # state.cfv_structure

        time_stamp = state.output[-1].time_stamp
        date_processing = time_stamp.replace('@', '-').replace(':', '-').split('-')
        date_processing = [int(v) for v in date_processing]
        prev_time_stamp = datetime(*date_processing)

        dt = (cur_time_stamp - prev_time_stamp).microseconds/1000000
        cfv = []
        cfw = []
        for function in range(len(cfv_structure)):
            cf_id = cfv_structure[function].id
            cf_params = cfv_structure[function].params
            cf_weight = cfv_structure[function].impactweight
            ### calling the library functions
            cf_impactvalue = computeCFValue(cf_id, cf_params, singleimpactmatrix, curriteration, dt)
            cfw.append(cf_weight)
            cfv.append(cf_impactvalue)

        """ aggimpact(j, k) = dot(cfw(j, :, k), cfv(j, :, k))/sum(cfw(j, :, k));"""

        aggimpact = np.dot(cfv, cfw)  # /sum(cfw)

        """ X(j, k + 1) = X(j, k) + s(j, k) * (aggimpact(j, k) - X(j, k)) * dt"""

        currentStateValue = state.getOutputValues()#curriteration)
        nextStateValue = currentStateValue + speed * (aggimpact - currentStateValue) * dt

        # setting Output


        date_time = cur_time_stamp.strftime("%Y-%m-%d@%H:%M:%S")

        initial_state = state.isInitialState()
        if(np.isnan(nextStateValue)):   #if the results returned NaN then set the value as the last value received
            nextStateValue = state.getOutputValues()
        soutput = StateOutput(date_time, nextStateValue)
        state.setOutputValues(soutput)
        state.setFinalValue(nextStateValue)
        state.setCompletionStatus(completionstatus)
        if index == 23 or index == 15:  # id
            print('state index,name', index, state.name, state.complete)
        x = 10
    x = 10


## support functions

def getCompletionStatus(completionstatusarr):
    if len(completionstatusarr) > 0:
        if False in completionstatusarr:
            return False
        else:
            return True
    else:
        return 'Unknown'

def is_valid_float(element: str) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False

def getStateOutputValues(connection, statematrix):
    #this function retuens the last values
    index = int(re.findall(r'\d+', connection)[0])
    state = statematrix[index - 1]
    outputvalue = state.getOutputValues(-1)
    # print('state, curr iter, connection ', state.index, curriteration, connection, outputvalue)

    return outputvalue


def getIncomingCompletionStatus(connection, statematrix,curstate):

    index = int(re.findall(r'\d+', connection)[0])
    state = statematrix[index - 1]
    completionstatus = state.complete
    return completionstatus


def findAdaptiveSpeed(speed):
    return re.findall(r'\d+', speed)[0]
def findAdaptiveState(astate):
    return re.findall(r'\d+', astate)[0]

####SimulationResults####

def extractId(id):
    id_array = re.findall(r'\d+', id)
    return int(id_array[0])
def extractValue(value):
    value = re.findall(r'[-+]?(?:\d*\.*\d+)', value)
    return float(value[0])
def extractIncomingConnectionWeights(state,connection_weights, in_connections):
    # input: list of dictionaries
    # returns matrix that constains incoming states
    # avoid self reference
    state_id = extractId(state['id'])
    if (state_id == 9):
        x = 10
    in_connection_list = []
    for idx,item in enumerate(in_connections):
        value = item['value']
        connection_id = extractId(value)
        if (connection_id != state_id):
            value = connection_weights[idx]['value']
            if (value != ''):
                value = extractValue(value)
            else:
                value = 0.0
            in_connection_list.append(value)
    return in_connection_list
def extractIncomingConnections(state,in_connections):
    # input: list of dictionaries
    # returns matrix that constains incoming states
    # avoid self reference
    state_id = extractId(state['id'])
    if(state_id == 9):
        x = 10
    in_connection_list = []
    for item in in_connections:
        value = item['value']
        connection_id = extractId(value)
        if(connection_id != state_id):
             if(value != ''):
                value = value #extractId(value)
             else:
                value = 0
             in_connection_list.append(value)
    return in_connection_list
def extractCFVStructure(combination_functions):
    #input: list
    # matrix using combination function
    cfv_structure = []
    for combinationfunction in combination_functions:
        id = int(combinationfunction['id'])
        name = combinationfunction['name']
        impactweight = float(combinationfunction['function_weight'])
        #print('PARAMS', combinationfunction['number_of_possible_params'], type(combinationfunction['number_of_possible_params']))
        #numberOfParams = int(combinationfunction['number_of_possible_params'])
        numberOfParams = combinationfunction['number_of_possible_params']
        parameters = combinationfunction['parameters']
        params = []
        for parameter in parameters:
            name = parameter['name']
            value = parameter['value']
            if (value != ''):
                value = float(value)
            else:
                value = 0
            params.append(value)

        cfv = CombiationFunctionStructure(id, name, numberOfParams, impactweight)
        cfv.setParams(params)
        cfv_structure.append(cfv)

    return cfv_structure



####################Controller methods


########Simulation Software Used################

######## ASSUMPTIONS

THRESHOLD = 0#.1

#########
class CombiationFunctionStructure:
    # matrix using combination function
    # [structure:{
    # combinationfunction: name,
    # impactweight: weight value,
    # parameters:{param1,param2}
    # },
    # ]
    def __init__(self, id, name, noparams, impactweight):
        self.id = id  # id of the function
        self.name = name  # what is the name of the function
        self.impactweight = impactweight  # what is the percentage of the function, if NaN then this function is not used
        self.numberOfParams = noparams  # set initially that how many parameters of this function are possible
        self.params = []  # array of parameters, length should be equal to numberOfParams

    def setParams(self, params):
        self.params = params

    # remove this
    def getNumberOfPossibleParams(self):
        return self.numberOfParams

    def setWeight(self, impactweight):
        self.impactweight = impactweight

class State:
    def __init__(self, id, index, name, b, cw,s,init_value,cfv_structure,level,complete,is_internal):#,initial_state):
        self.index = index
        self.id = id
        self.name = name  # string
        self.incomingconnections = b  # matrix that takes states (in a row) as input
        self.initialvalue = init_value  # initial value int/float
        self.finalvalue = 0    # final value int/float
        self.connection_weights = cw  # matrix that takes connection weights (in a row) length should be equal to incoming states
        self.cfv_structure = cfv_structure  # matrix using combination function
        self.output = []
        self.mcfw = []
        self.speed = s
        self.level = level
        self.complete = complete
        #self.initial_state = initial_state
        self.is_internal = is_internal


    def isInitialState(self):
        if (self.incomingconnections):
            incominglist = self.incomingconnections
            if(len(incominglist) == 1 and incominglist[0] != self.id):
                return False
            else:
                return True
        else:
            return False
    def setOutputValues(self, output):
        self.output.append(output)

    def getOutputValues(self, iterationvalue = -1):
        return self.output[iterationvalue].output

    def getLastOutput(self, iterationvalue = -1):
        return self.output[iterationvalue]

    def getLevel(self):
        return self.level

    def getid(self):
        return self.id
    def setSpeedFactor(self, speed):
        self.speed = speed

    def setConnectionWeight(self, mcw):
        self.connection_weights = mcw  # matrix that takes connection weights (in a row)

    def setCombinationFunctionStructure(self, cf):
        self.cfv_structure = cf

    def setInitialValue(self, iv):
        self.initialvalue = iv
    def setFinalValue(self, fv):
        self.finalvalue = fv

    def setCompletionStatus(self, completionstatus):
       self.complete = completionstatus

class StateOutput:
    def __init__(self, t, outputvalue):
        self.time_stamp = t  # time stamp of output per state
        self.output = outputvalue  # value of the state

    def getValue(self):
        return self.output

    def getTimeStamp(self):
        return self.time_stamp

############################### Start Library Code
def normalizeValue(value):
    # this function is used to normalize the value between 0 and 1
    if value >= 0:
        ans = value
    else:
        ans = 0

    if ans <= 0:
        ans = 0

    return ans
def eucl(params, values):
    """Euclidean
    params = order n, scaling factor
    """
    order = params[0]
    scalingfactor = params[1]
    # y = (sum(v.^p(1))/p(2)).^(1/p(1));
    ans = ((values ** order).sum(axis=0) / scalingfactor) ** (1 / order)
    # print('eucledian ans', ans)
    return ans
def alogistic(params, values):
    """ advanced logistic sum
    params = P(0) steepness, P(1)threshold
    y = (1/(1+exp(-p(1)*(sum(v)-p(2))))-1/(1+exp(p(1)*p(2))))*(1+exp(-p(1)*p(2)));
    x=max(0,y);
    """

    steepness = params[0]
    threshold = params[1]
    sumv = values.sum(axis=0)
    power = -abs(steepness) * (sumv - threshold)
    ans = ((1 / (1 + math.exp(power))) - (1 / (1 + math.exp(steepness * threshold)))) * (
                1 + math.exp((-abs(steepness)) * threshold))
    ans = normalizeValue(ans)

    return ans
def hebb(params, values):
    """Hebbian learning
    p(1) = persistence factor """
    source = values[0]  # v1
    destination = values[1]  # v2
    wstate = values[2]  # w
    persistence = params[0]

    return source * destination * (1 - wstate) + persistence * wstate
def scm(params, values):
    """ state-connection modulation
    p(1) = modulation factor
    """
    v1 = values[0]
    w = values[1]
    modulation = params[0]

    return w + (modulation * v1 * w * (1 - w))
def slhomo(params, values):
    """simple linear homophily
    params = amplification factor, tipping point
    """
    v1 = values[0]
    v2 = values[1]
    w = values[2]
    amplification = params[0]
    tippingpoint = params[1]

    return w + (amplification * (tippingpoint - abs(v1 - v2) * (1 - w) * w))
def sqhomo(params, values):
    """simple quadratic homophily
    params = amplification factor , tipping point 
    """
    v1 = values[0]
    v2 = values[1]
    w = values[2]
    amplification = params[0]
    tippingpoint = params[1]

    return w + (amplification * w * (1 - w) * (pow(tippingpoint, 2) - pow((v1 - v2), 2)))
def alhomo(params, values):
    """advanced linear homophily
    params = amplification factor, tipping point

    """
    v1 = values[0]
    v2 = values[1]
    w = values[2]
    amplification = params[0]
    tippingpoint = params[1]

    x = amplification * (tippingpoint - abs(v1 - v2))

    if x > 0:
        nx = -x
        posx = (abs(x) + x) / 2
        posnx = (abs(nx) + nx) / 2

        return w + (posx * (1 - w)) - (posnx * w)
    else:
        return 0
def aqhomo(params, values):
    """advanced quadratic homophily
    params = amplification factor, tipping point
    values =
    """
    v1 = values[0]
    v2 = values[1]
    w = values[2]
    amplification = params[0]
    tippingpoint = params[1]

    x = amplification * (pow(tippingpoint, 2) - pow(abs(v1 - v2), 2))

    if x > 0:
        nx = -x
        posx = (abs(x) + x) / 2
        posnx = (abs(nx) + nx) / 2

        return w + (posx * (1 - w)) - (posnx * w)
    else:
        return 0
def sconnhebb(params, values):
    """squared connection hebbian learning
    p(1) = persistence factor
    """

    source = values[0]  # v1
    destination = values[1]  # v2
    wstate = values[2]
    persistance = params[0]

    return (source * destination * (1 - pow(wstate, 2))) + (persistance * wstate)
def srconnhebb(params, values):
    """square-root connection hebbian learning
    params = persistence factor
    """
    source = values[0]
    destination = values[1]
    wstate = values[2]
    persistance = params[0]

    return (source * destination * (1 - math.sqrt(wstate))) + (persistance * wstate)
def srstateshebb(params, values):
    """
    square-root state values hebbian learning
    :param params: persistence factor
    :param values:
    :return:
    """
    source = values[0]
    destination = values[1]
    wstate = values[2]
    persistance = params[0]

    return (math.sqrt(source * destination) * (1 - math.sqrt(wstate))) + persistance * wstate
def sstateshebb(params, values):
    """
    squared state values hebbian learning
    :param params: persistence factor
    :param values:
    :return:
    """

    source = values[0]
    destination = values[1]
    wstate = values[2]
    persistance = params[0]

    return (pow(source * destination, 2) * (1 - wstate)) + persistance * wstate
def slogistic(params, values):
    """
    simple logistic sum
    :param params: steepness, threshold
    :param values:
    :return:
    """

    steepness = params[0]
    threshold = params[1]
    sumv = values.sum(axis=0)
    power = -abs(steepness) * (sumv - threshold)
    ans = 1 / (1 + math.exp(power))
    ans = normalizeValue(ans)

    return ans
def cubehomo(params, values):
    """
    cubic homophily
    :param params: amplification factor , tipping point
    :param values:
    :return:
    """
    v1 = values[0]
    v2 = values[1]
    w = values[2]
    amplification = params[0]
    tippingpoint = params[1]

    return w + amplification * (1 - w) * pow((1 - abs(v1 - v2) / tippingpoint), 3)
def exphomo(params, values):
    """
    exponential homophily
    p(1) = steepness 
    p(2) = tipping point 
    """
    v1 = values[0]
    v2 = values[1]
    w = values[2]
    steepness = params[0]
    tippingpoint = params[1]

    return 1 - (1 - w) * math.exp(steepness * (abs(v1 - v2) - tippingpoint))
def log1homo(params, values):
    """logistic homophily 1
        p(1) = steepness 
        p(2) = tipping point """
    v1 = values[0]
    v2 = values[1]
    w = values[2]
    steepness = params[0]
    tippingpoint = params[1]

    return w / (w + (1 - w) * math.exp(steepness * abs(v1 - v2) - tippingpoint))
def log2homo(params, values):
    """logistic homophily 2
    p(1) = amplification factor 
    p(2) = steepness 
    p(3) = tipping point"""
    v1 = values[0]
    v2 = values[1]
    w = values[2]
    amplification = params[0]
    steepness = params[1]
    tipping = params[2]
    power = - steepness * (abs(v1 - v2) - tipping)

    return w + amplification * ((w * (1 - w)) / (1 + math.exp(power)))
def sinhomo(params, values):
    """sinus homophily
    p(1) = amplification factor 
    p(2) = tipping point"""
    v1 = values[0]
    v2 = values[1]
    w = values[2]
    amplification = params[0]
    tippingpoint = params[1]

    return w - amplification * (1 - w) * math.sin(math.pi * (abs(v1 - v2) - tippingpoint) / 2)
def tanhomo(params, values):
    """tangent homophily
    p(1) = amplification factor 
    p(2) = tipping point """

    v1 = values[0]
    v2 = values[1]
    w = values[2]
    amplification = params[0]
    tippingpoint = params[1]

    return w - amplification * (1 - w) * math.tan(math.pi * (abs(v1 - v2) - tippingpoint) / 2)
def invtan(params, values):
    """scaled inverse tangent sum
    p(1) = scaling factor """

    scalingfactor = params[0]
    sumv = values.sum(axis=0)

    return 2 * math.atan(sumv / scalingfactor) / math.pi
def id(params, values):
    """identity
    No parameter"""
    return values[0]
def complementid(params, values):
    """complement identity
    No parameter"""
    return (1 - values[0])
def product(params, values):
    """product
    No parameter"""
    return np.product(values)
def coproduct(params, values):
    """coproduct
    No parameter"""

    return 1 - np.product(1 - values)
def sminimum(params, values):
    """scaled minimum
    p(1) = scaling factor """

    scaling = params[0]
    return np.min(values) / scaling
def smaximum(params, values):
    """scaled maximum
    p(1) = scaling factor """
    scaling = params[0]
    return np.max(values) / scaling
def aproduct(params, values):
    """advanced product
    p(1) = weight """

    weight = params[0]
    return weight * coproduct(params, values) + (1 - weight) * product(params, values)
def aminmax(params, values):
    """advanced min-max
    p(1) = weight """
    weight = params[0]
    return weight * np.max(values) + (1 - weight) * np.min(values)
def multicriteriahomo(params, values):
    """multicriteria homophily
    p(1) = amplification factor 
    p(2) = tipping point """
    # TODO Check its implementation
    amplification = params[0]
    tippingpoint = params[1]
    w = values[-1]  # lastelem

    halves = np.array_split(values[0:len(values) - 1], 2)
    firsthalf = np.array(halves[0])
    secondhalf = np.array(halves[1])

    if len(firsthalf) == len(secondhalf):
        dissimilarity = math.sqrt(sum(pow((secondhalf - firsthalf), 2)))
    else:
        if len(firsthalf) > len(secondhalf):
            array = halves[0]
            firsthalf = array[0:len(array) - 1]
            secondhalf = np.array(halves[1])
        else:
            firsthalf = np.array(halves[0])
            array = halves[1]
            secondhalf = array[0:len(array) - 1]
        dissimilarity = math.sqrt(sum(pow((secondhalf - firsthalf), 2)))

    return w + amplification * w * (1 - w) * (tippingpoint - dissimilarity)
def ssum(params, values):
    """scaled sum
    p(1) = scaling factor """
    scalingfactor = params[0]
    sumv = values.sum(axis=0)

    return sumv / scalingfactor
def adnormsum(params, values):
    """advanced normalised sum
    No parameter"""

    halves = np.array_split(values[0:len(values) - 1], 2)
    firsthalf = np.array(halves[0])
    secondhalf = np.array(halves[1])

    return ssum([sum(firsthalf)], secondhalf)
def adnormeucl(params, values):
    """advanced normalised Euclidean
    p(1) = order n"""
    # TODO Check its implementation Discuss with Jan
    order = params[0]
    halves = np.array_split(values[0:len(values)], 2)
    firsthalf = np.array(halves[0])
    firsthalfpowers = pow(firsthalf, order)
    secondhalf = np.array(halves[1])  # TODO: shouldnt power be taken here?

    return eucl([order, sum(firsthalfpowers)], secondhalf)
def sgeomean(params, values):
    """scaled geometric mean
    p(1) = scaling factor"""
    # OK
    scalingfactor = params[0]
    nonzeroelems = values[values != 0]

    return (product(1, nonzeroelems) / scalingfactor) ** (1 / nonzeroelems.size)
def s2scm(params, values):
    """state-connection modulation shifted
    p(1) = modulation factor """
    modulation = params[0]
    cweight_w = values[2]
    value = values[3]
    return cweight_w + modulation * value * cweight_w * (1 - cweight_w)
def stepmod(params, values, curriteration, dt):
    """step-modulo
    p(1) = repeated time duration - t repeats this behavior until this duration
    p(2) = tipping point - where to change it"""

    timeduration = params[0]
    tippingpoint = params[1]
    current = curriteration * dt

    modvalue = current % timeduration
    if modvalue < tippingpoint:
        ans = 0
    else:
        if modvalue >= tippingpoint:
            ans = 1

    return ans
def stepmodopp(params, values, curriteration, dt):
    """opposite step-modulo
    p(1) = repeated time duration 
    p(2) = tipping point """

    timeduration = params[0]
    tippingpoint = params[1]
    current = curriteration * dt

    modvalue = current % timeduration
    if modvalue < tippingpoint:
        ans = 1
    else:
        if modvalue >= tippingpoint:
            ans = 0

    return ans
def hebbneg(params, values):
    """hebbian learning for negative weights
    -V1(1-V2 )(1+W) + W
    p(1) = persistence factor """
    persistancefactor = params[0]
    source = values[0]
    destination = values[1]
    wstate = values[2]

    return -(source * (1 - destination) * (1 + wstate)) + persistancefactor * wstate
def sigmoid(params, values):
    """sigmoid sum function
    p(1) = steepness 
    p(2) = threshold """

    steepness = params[0]
    threshold = params[1]
    sumv = values.sum(axis=0)

    return pow(sumv, steepness) / (pow(sumv, steepness) + pow(threshold, steepness))
def hebbqual(params, values):
    """qualitative hebbian learning
    p(1) = persistence factor """

    persistancefactor = params[0]
    source = values[0]
    destination = values[1]
    wstate = values[2]
    if source > 0.5:
        source = 1.0
    else:
        source = 0.0

    if destination > 0.5:
        destination = 1.0
    else:
        destination = 0.0

    return source * destination * (1 - wstate) + persistancefactor * wstate
def bcfvalues(nrs, ps, vs, ks, curriteration, dt):
    x = []
    for n in range(len(nrs)):
        startindex = 2 * n
        endindex = 2 * n + 1
        p = [ps[index] for index in range(startindex, endindex + 1)]
        v = np.array(vs[0])
        if n == 0:
            v = np.array([vs[index] for index in range(ks[0])])
        else:
            if n > 0:
                array = ks[0:len(ks) - 1]
                sumks_n = ks.sum(axis=0)
                sumks_n_1 = array.sum(axis=0)

                v = np.array([vs[index] for index in range(sumks_n_1 + 1, sumks_n)])

        x.append(computeCFValue(nrs[n], p, v, curriteration, dt))
    return x
def composedbcfs(h, p, nrs, ps, vs, ks, curriteration, dt):
    return computeCFValue(h, p, bcfvalues(nrs, ps, vs, ks, curriteration, dt), curriteration, dt)
def max2and1bcfs(params, values, curriteration, dt):
    """
    This function is composed of three functions from library. It takes maximum (smax) of eucl and alogistic
    :param params: p(1) and p(2) are the numbers of the bcf's composed by the max operator;
    each of them has fixed parameters [1 1] and the first one takes 2 values and the second one takes 1 value.:

    """
    return composedbcfs(26, np.array([1, 1]), params, np.array([1, 1, 1, 1]), values, np.array([2, 1]), curriteration,
                        dt)
def boundedgrowth(params, values):
    """bounded growth by Verhulst’s classical model
    p(1) = carrying capacity """
    capacity = params[0]
    sumv = values.sum(axis=0)

    return (2 - sumv / capacity) * sumv
def min_alog_composition(params, values, curriteration, dt):
    """Composition of alogistic,(..) and id(.) function by minimum function smin1(..)
    p(1) = steepness 
    p(2) = threshold """
    # TODO: Check the implementation
    steepness = params[0]
    threshold = params[1]

    return composedbcfs(25, np.array([1, 1]), np.array([2, 21]), np.array([steepness, threshold, 0, 0]), values,
                        np.array([7, 1]), curriteration, dt)
def steponce(params, values, curriteration, dt):
    """one period of activation from  top 
    p(1) = start time 
    p(2) = end time """
    starttime = params[0]
    endtime = params[1]

    currenttime = curriteration * dt

    if currenttime >= starttime and currenttime <= endtime:
        return 1
    else:
        return 0
def scalemap(params, values):
    """mapping activation scale [0, 1] to scale [, ]
    p(1) = lower bound 
    p(2) = upper bound """
    upperbound = params[0]
    lowerbound = params[1]

    return lowerbound + (upperbound - lowerbound) * values[0]
def scalemapopp(params, values):
    """opposite mapping activation scale [0, 1] to scale [, ]]
    p(1) = lower bound 
    p(2) = upper bound """
    upperbound = params[0]
    lowerbound = params[1]

    return lowerbound - (upperbound - lowerbound) * (values[0])
def posdev(params, values):
    """positive deviation of V from norm 
    p(1) = norm """

    norm = params[0]

    if values[0] >= norm:
        return values[0] - norm
    else:
        return 0
def negdev(params, values):
    """negative deviation of V from norm 
    p(1) = norm """
    norm = params[0]

    if values[0] < norm:
        return values[0] - norm
    else:
        return 0
def steps(params, values, curriteration, dt):
    """"""
    timeduration = params[0]
    tippingpoint = params[1]
    currenttime = curriteration * dt

    if currenttime < timeduration:
        ans = 405
    else:
        if currenttime >= timeduration and currenttime <= tippingpoint:
            ans = 380
        else:
            if currenttime > tippingpoint:
                ans = 399

    return ans
def maxhebb(params, values):
    """ TODO NOT COMPLETE
    :param params:
    :param values:
    :return:
    """

    hebbarray = values[0:3]
    nonhebarray = values[3:len(values)]
    hebvalue = hebb(params, hebbarray)

    nparray = np.append(hebvalue, nonhebarray)
    return np.max(nparray)
def maxmin2(params, values):
    """ TODO NOT COMPLETE
    :param params:
    :param values:
    :return:
    """
    firsthalfindex = params[0]
    secondhalfindex = params[1]

    firsthalf = values[0:firsthalfindex]
    secondhalf = values[firsthalfindex: firsthalfindex + secondhalfindex]
    return np.max(np.min(firsthalf), np.min(secondhalf))
def maxmin3(params, values):
    """ TODO NOT COMPLETE
    :param params:
    :param values:
    :return:
    """
    firsthalfindex = params[0]
    secondhalfindex = params[1]

    firstgroup = values[0:firsthalfindex]
    secondgroup = values[firsthalfindex: firsthalfindex + secondhalfindex]
    lastgroup = values[firsthalfindex + secondhalfindex: len(values)]
    minarray = []
    minarray.append(np.min(firstgroup))
    minarray.append(np.min(secondgroup))
    minarray.append(np.min(lastgroup))
    return np.max(minarray)
def randsteponce(params, values, curriteration, dt):
    """
    randomly sets 0 and 1
    :param params: start p(0) and end p(1) time
    :param values: any integer values
    :return: a random number from 0 to 1
    """
    start = params[0]
    end = params[1]

    current = curriteration * dt

    if current >= start and current <= end:
        return np.random.rand(1, 1)[0][0]
    else:
        if current < start or current > end:
            return 0
def randstepmod(params, values, curriteration, dt):
    """
    :param params: time duration & tipping point & persistance & lower bound
    :param values: time duration (any) & tipping point (0-1) & persistance (0-1) & lower bound ()
    :param curriteration:
    :param dt:
    :return: 0 to 1 values
    """
    timeduration = params[0]
    tippingpoint = params[1]

    # TODO: set these values from input Parameters
    persistance = 0
    lbound = 0.5
    random = np.random.rand(1, 1)[0][0]

    current = curriteration * dt

    modvalue = current % timeduration

    if modvalue < tippingpoint:
        ans = 0
    else:
        if modvalue >= tippingpoint:
            ans = persistance * values[0] + (1 - persistance) * (lbound + random * (1 - lbound))

    return ans
def compdiff(params, values):
    """
    Description
    :param params: values have two series (v1, v2)
    :param values:
    :return:
    """
    max_val = max(values[0], values[1])  # maximum of two lists
    if max_val > 0:
        ans = 1 - abs(values[0] - values[1]) / max_val
    else:
        ans = 0
    return ans
def randstepmodopp(params, values,curriteration,dt):
    """
        :param params: time duration & tipping point & persistance & lower bound
        :param values: time duration (any) & tipping point (0-1) & persistance (0-1) & lower bound ()
        :param curriteration:
        :param dt:
        :return: 0 to 1 values
        """
    timeduration = params[0]
    tippingpoint = params[1]

    # TODO: set these values from input Parameters
    persistance = params[2]  # 0
    lbound = params[3]  # 0.5
    random = np.random.rand(1, 1)[0][0]

    current = curriteration * dt

    modvalue = current % timeduration

    if modvalue >= tippingpoint:
        ans = 0
    else:
        if modvalue < tippingpoint:
            ans = persistance * values[0] + (1 - persistance) * (lbound + random * (1 - lbound))

    return ans
def swcorrcoeff(params, values):
    ################################################################################################
    return 0
def randstepmodalt(params, values, curriteration, dt):
    """
            :param params: repeated time duration & durationuntil1 & persistance & lower bound
            :param values: time duration (any) & tipping point (0-1) & persistance (0-1) & lower bound ()
            :param curriteration: k
            :param dt:
            :return: 0 to 1 values
            """
    timeduration = params[0]
    durationuntil1 = params[1]

    # TODO: set these values from input Parameters
    persistance = params[2]  # 0
    lbound = params[3]  # 0.5

    random = np.random.rand(1, 1)[0][0]

    current = curriteration * dt

    modvalue = current % timeduration

    if modvalue < durationuntil1:
        ans = 0
    else:
        if modvalue >= durationuntil1:
            ans = persistance * values[0] + (1 - persistance) * (lbound + random * (1 - lbound))

    return ans
def compdiffnorm(params, values, curriteration, dt):
    """

    :param params: p(1): id of the state, v(1) and v(2) current values of two series
    :param values: p(1):N, 0-1
    :return: 0/0-1
    """
    # sumseries1 =
    # sumseries2 =

    # stateid = params[0]

    return 0
def swcorrcoeff0(params, values):
    ################################################################################################
    return 0
def compdifflag(params, values):
    ################################################################################################
    return 0
def compdifflag1av(params, values):
    ################################################################################################
    return 0
def compdifflag2av(params, values):
    ################################################################################################
    return 0
def swcorrcoeffsu(params, values):
    ################################################################################################
    return 0
def transdetav(params, values):
    return 0
def transdetmaxmin(params, values):
    return 0
def transabs(params, values):
    return 0
def transdetavabs(params, values):
    return 0
def transdetstdev(params, values):
    return 0
def monitor(params, values):
    '''
    if v(1) - v(2) >= p(1)  #HCP didnt do well
        x=1
        else if v(1) - v(2) < p(1)  #he did well
            x = 0
            '''
    v1 = values[0]
    v2 = values[1]
    threshold = params[0]

    if v1 - v2 >= threshold:
        return 1
    else:
        return 0
def default(params, values):
    return id(params, values)
def computeCFValue(noOfcf, parameters, values, curriteration, dt):
    # this is the library of combination functions termed as bcf in matlab
    switcher = {
        1: (eucl, (parameters, values)),
        2: (alogistic, (parameters, values)),
        3: (hebb, (parameters, values)),
        4: (scm, (parameters, values)),
        5: (slhomo, (parameters, values)),
        6: (sqhomo, (parameters, values)),
        7: (alhomo, (parameters, values)),
        8: (aqhomo, (parameters, values)),
        9: (sconnhebb, (parameters, values)),
        10: (srconnhebb, (parameters, values)),
        11: (srstateshebb, (parameters, values)),
        12: (sstateshebb, (parameters, values)),
        13: (slogistic, (parameters, values)),
        14: (cubehomo, (parameters, values)),
        15: (exphomo, (parameters, values)),
        16: (log1homo, (parameters, values)),
        17: (log2homo, (parameters, values)),
        18: (sinhomo, (parameters, values)),
        19: (tanhomo, (parameters, values)),
        20: (invtan, (parameters, values)),
        21: (id, (parameters, values)),
        22: (complementid, (parameters, values)),
        23: (product, (parameters, values)),
        24: (coproduct, (parameters, values)),
        25: (sminimum, (parameters, values)),
        26: (smaximum, (parameters, values)),
        27: (aproduct, (parameters, values)),
        28: (aminmax, (parameters, values)),
        29: (multicriteriahomo, (parameters, values)),
        30: (ssum, (parameters, values)),
        31: (adnormsum, (parameters, values)),
        32: (adnormeucl, (parameters, values)),
        33: (sgeomean, (parameters, values)),
        34: (s2scm, (parameters, values)),
        35: (stepmod, (parameters, values, curriteration, dt)),
        36: (stepmodopp, (parameters, values, curriteration, dt)),
        37: (hebbneg, (parameters, values)),
        38: (sigmoid, (parameters, values)),
        39: (hebbqual, (parameters, values)),
        40: (max2and1bcfs, (parameters, values, curriteration, dt)),
        41: (boundedgrowth, (parameters, values)),
        42: (min_alog_composition, (parameters, values, curriteration, dt)),
        43: (steponce, (parameters, values, curriteration, dt)),
        44: (scalemap, (parameters, values)),
        45: (scalemapopp, (parameters, values)),
        46: (posdev, (parameters, values)),
        47: (negdev, (parameters, values)),
        48: (steps, (parameters, values)),
        49: (maxhebb, (parameters, values)),
        50: (maxmin2, (parameters, values)),
        51: (maxmin3, (parameters, values)),
        52: (randsteponce, (parameters, values,curriteration,dt)),
        53: (randstepmod, (parameters, values)),
        54: (compdiff, (parameters, values)),
        55: (randstepmodopp, (parameters, values,curriteration,dt)),
        56: (swcorrcoeff, (parameters, values)),
        57: (randstepmodalt, (parameters, values)),
        58: (compdiffnorm, (parameters, values)),
        59: (swcorrcoeff0, (parameters, values)),
        60: (compdifflag, (parameters, values)),
        61: (compdifflag1av, (parameters, values)),
        62: (compdifflag2av, (parameters, values)),
        63: (swcorrcoeffsu, (parameters, values)),
        64: (transdetav, (parameters, values)),
        65: (transdetmaxmin, (parameters, values)),
        66: (transabs, (parameters, values)),
        67: (transdetavabs, (parameters, values)),
        68: (transdetstdev, (parameters, values)),
        69: (monitor, (parameters, values))

    }
    func, args = switcher.get(noOfcf, (None, None))
    if func is not None:
        return func(*args)
    # return switcher.get(noOfcf,20000)#default(parameters,values))


############################### End Library Code
