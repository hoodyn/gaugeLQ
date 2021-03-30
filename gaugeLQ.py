import smelli
from ULR2WC import ULR2WC
from wilson import Wilson
import numpy as np
from time import perf_counter
gl = smelli.GlobalLikelihood(exclude_likelihoods = ['likelihood_zlfv.yaml'])
Wilson.set_default_option('smeft_accuracy', 'integrate')

GeV = 1.
TeV = 1000*GeV

import pickle
pickle_dir = 'datasets/'
log_dir = 'logs/'

def depicklit(filename, dirname = pickle_dir):
    with open(dirname + filename, 'rb') as f:
        return pickle.load(f)

def picklit(dataset, filename, dirname = pickle_dir):
    with open(dirname + filename, 'wb') as f:
        pickle.dump(dataset, f)

def makeWilson(pt_dic, mLQ = 'mLQ_WM'):
    """Returns Wilson instance, provided with a parameter point
    in our conventions.
    LQ mass can be specified as a float or as a string key in pt_dic.
    """
    UL = np.array(pt_dic['UL'])
    UR = np.array(pt_dic['UR'])
    if isinstance(mLQ, str):
        _mLQ = pt_dic[mLQ]
    elif isinstance(mLQ, float):
        _mLQ = mLQ
    WCs = ULR2WC(UL, UR, _mLQ)
    wil = Wilson(WCs, scale=_mLQ, eft='SMEFT', basis='Warsaw')
    return wil

SM_parameter_point = gl.parameter_point({}, scale=0.1*TeV)
#SM_obstable = SM_parameter_point.obstable()
SM_obstable = pickle.load(open('sm_obstable.pickle','rb'))
SM_logL = SM_parameter_point.log_likelihood_global()
SM_logL

def anomalous_observables(obstable, pull_min = 2.5):
    obstable_head = obstable[obstable['pull exp.'] > pull_min]
    return tuple(obstable_head.index)

def SM_anomalous_observables(pull_min_SM = 2.5):
    return anomalous_observables(SM_obstable, pull_min_SM)

def anomalous_observables_new(obstable, pull_min, pull_min_SM):
    signals = anomalous_observables(obstable, pull_min)
    SM_signals = SM_anomalous_observables(pull_min_SM)
    return tuple(obs for obs in signals if obs not in SM_signals)

def anomalous_observables_explained(obstable, pull_min, pull_min_SM):
    signals = anomalous_observables(obstable, pull_min)
    SM_signals = SM_anomalous_observables(pull_min_SM)
    return tuple(obs for obs in SM_signals if obs not in signals)

#likelihoods = [*gl._fast_likelihoods_yaml, *gl._likelihoods_yaml]

default_policy = {'catastrophic_logL': -15.,
                  'too_bad_logL': -8., 
                  'bad_logL': -3.,
                  'good_logL': +2.,
                  'pull_min': 2.5,
                  'pull_min_SM': 2.5
                 }

extra_obs_default = {'BR(KS->emu,mue)'}

def analyzeDataPoint(datapoint, policy = default_policy, extra_obs_to_calculate = extra_obs_default, 
                     message = print, log_policy = False):
    """Function to analyze given shape of quark-lepton mixing matrices.
    """
    t0 = perf_counter()
    message('\nWorking on new dataPoint.')
    if log_policy == True:
        message('Parameters of the analyzis:', policy)

    mLQ = datapoint['mLQ_WM']
    shift_mLQ = False  # A flag saying how to shift the mLQ ('up' or 'down')
    tested_mLQs = []   # List of all tested mLQ. Just for tuning.
    
    while 'firstSignal' not in datapoint:
        if shift_mLQ == 'down':
            mLQ *= 0.9
        elif shift_mLQ == 'up':
            mLQ *= 1.2
        elif shift_mLQ == 'way_up':
            mLQ *= 1.8
        tested_mLQs.append(mLQ)
        message('Studying mLQ =', mLQ, 'GeV.')
        
        glp = gl.parameter_point(makeWilson(datapoint, mLQ))
        logL = glp.log_likelihood_global()
        message('logL = ', logL)
        
        if (logL < policy['catastrophic_logL']):
            message('Catastrophic likelihood...')
            shift_mLQ = 'way_up'   # If too strong bad signal,
            continue               # try again with a heavier LQ mass.
        
        if (policy['catastrophic_logL'] < logL < policy['too_bad_logL']):
            message('Too bad likelihood...')
            shift_mLQ = 'up'   # If too strong bad signal,
            continue           # try again with a heavier LQ mass.
        
        obstable = glp.obstable() # Improve this line! Calculate just relevant part of it.
        new_observables = anomalous_observables_new(obstable, policy['pull_min'], policy['pull_min_SM'])
        
        if  len(new_observables) > 1:
            message('Multiple new observables:', new_observables)
            shift_mLQ = 'up'
            continue

        if (policy['bad_logL'] < logL < policy['good_logL']) and len(new_observables) == 0:
            message('No signal.')
            shift_mLQ = 'down'
            continue
        
        if policy['good_logL'] < logL: 
            datapoint['firstSignal'] = 'Improving current likelihood!'
            message(datapoint['firstSignal'])
            continue
        elif len(new_observables) == 1:
            datapoint['firstSignal'] = new_observables[0]
            message('First signal:', datapoint['firstSignal'])
            continue
        elif policy['too_bad_logL'] < logL < policy['bad_logL']:
            datapoint['firstSignal'] = 'Future set of anomalies'
            message(datapoint['firstSignal'])
            continue
        raise AssertionError("We're where we shouldn't be...")
        
    datapoint['mLQ'] = mLQ
    datapoint['obstable'] = obstable
    datapoint['logL'] = logL
    datapoint['likelihoods'] = glp.log_likelihood_dict()
    
    extra_predicitons = {obs: smelli.flavio.np_prediction(obs, glp.w) 
                         for obs in extra_obs_to_calculate}
    datapoint['extra_predictions'] = extra_predicitons
    
    t1 = perf_counter()
    datapoint['analysis_metadata'] = {'tested_mLQs': tested_mLQs, 'time (s)': t1-t0}


def make_log_message_f(filename = 'log.log'):
    """Returns a function to be used instead of 'print'
    which prints on both the standard output 
    and to the log file (appends to the end)."""
    def log_message_f(*args):
        with open(filename,'a') as f:
            print(*args, file=f)
        print(*args)
    return log_message_f


def analyzeDataset(dataset, policy = default_policy, append_info = True, 
                  message = print, result_filename = 'dataset_analyzed.pickle',
                  extra_obs_to_calculate = extra_obs_default,
                  save_period = 10, update = False):
    message('Started working on new dataset:', dataset['info'])
    message('Number of points:', len(dataset['data']))
    message('Parameters of the analyzis:', policy)

    t0 = perf_counter()
    
    for (n,pt) in enumerate(dataset['data']):
        message('\n ==',n,'==')
        if ('firstSignal' in pt) and (update == False):
            continue
        else:
            analyzeDataPoint(pt, policy = policy, message = message, 
                            extra_obs_to_calculate = extra_obs_to_calculate)
            if (result_filename != False) and (n % save_period == 0):
                picklit(dataset, result_filename)
                message('The partially analyzed dataset saved to ', result_filename)


    t1 = perf_counter()    
    message('\nAnalysis of '+str(len(dataset['data']))+' points avoiding K0L took', (t1-t0)/60, 'minutes.') 
    # Append info not only to log_file opened by 'message'
    # but also to the dataset itself:
    if append_info is True:
        dataset['info'] = (dataset['info'], {'analyzed (minutes)': (t1-t0)/60,
                                             'policy': policy})
    elif append_info is not None:
        dataset['info'] = (dataset['info'], append_info)

    if result_filename != False:
        picklit(dataset, result_filename)
        message('The analyzed dataset saved to ', result_filename, '\n')


def analyzePickledDataset(input_filename, output_filename = True, log_filename = True,
                          extra_obs_to_calculate = extra_obs_default, save_period = 10,
                          update = False):
    if output_filename is True:
        _output_filename = input_filename + ".analyzed"
    else: # string
        _output_filename = output_filename
    
    if log_filename is True:
        _log_filename = log_dir +  input_filename + ".log"
    else: # string
        _log_filename = log_dir + log_filename
    
    log_message = make_log_message_f(filename=_log_filename)

    dataset = depicklit(input_filename)
    analyzeDataset(dataset, message = log_message, 
                   result_filename = _output_filename,
                   extra_obs_to_calculate = extra_obs_to_calculate,
                   save_period = save_period,
                   update = update)
    # picklit(dataset, filename = _output_filename) ... This is already inside 'analyzeDataset'
