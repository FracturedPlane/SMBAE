

from trainMetaModel import trainMetaModel, _trainMetaModel
import sys
import json
import copy
from pathos.threading import ThreadPool
from pathos.multiprocessing import ProcessingPool
import time
import datetime
from tools.PlotMetadataSimulation import plotMetaDataSimulation

from util.SimulationUtil import getDataDirectory, getBaseDataDirectory, getRootDataDirectory
"""
def tuneHyperParameters(simsettingsFileName, Hypersettings=None):
"""
        # For some set of parameters the functino will sample a number of them
        # In order to find a more optimal configuration.
"""
    import os
    num_sim_samples=5
    file = open(simsettingsFileName)
    settings = json.load(file)
    print ("Settings: " + str(json.dumps(settings, indent=4)))
    file.close()
    samples = 5
    param_of_interest = 'action_learning_rate'
    range_ = [0.05, 1.0]
    data_name = settings['data_folder']
    for i in range(samples+1):
        param_value = ((range_[1] - range_[0]) * (float(i)/samples)) + range_[0]
        settings['data_folder'] = data_name + "_" + str(param_value) + "/"
        settings['action_learning_rate'] = param_value
        directory= getBaseDataDirectory(settings)
        if not os.path.exists(directory):
            os.makedirs(directory)
        # file = open(settingsFileName, 'r')
        out_file_name=directory+os.path.basename(simsettingsFileName)
        
        print ("Saving settings file with data to: ", out_file_name)
        out_file = open(out_file_name, 'w')
        out_file.write(json.dumps(settings, indent=4))
        # file.close()
        out_file.close()
        
        trainMetaModel(simsettingsFileName, samples=num_sim_samples, settings=copy.deepcopy(settings), numThreads=num_sim_samples)
"""

def compute_next_val(range_,i,samples, curve_scheme='linear'):
    """
    
    """
    if ( (curve_scheme == 'linear')):
        delta_ = ((float(i)) / float(samples))
    elif (curve_scheme == "squared"):
        delta_ = ((float(i)) / float(samples))
        delta_ = delta_**2
    elif (curve_scheme == "exponential"):
        delta_ = ((float(i)) / float(samples))
        delta_ = delta_**(samples-i) 
        
    delta_ = delta_ * (range_[1] - range_[0]) 
    return delta_

def makeNiceName(params_to_tune):
    """
        Take the list of parameters to sample over and return a nice
        string that will result in a good filename.
    """
    out = ""
    for p in params_to_tune:
        out = out + "_" + str(p)
    return out

def get_param_values(hyper_settings):
    """
        Returns the cross product of the parameters
    """
    import itertools
    
    parameter_samples = hyper_settings['num_param_samples']
    params_ = []
    for par in range(len(parameter_samples)):
        param_of_interest = hyper_settings['param_to_tune'][par]
        range_ = hyper_settings['param_bounds'][par]
        samples = hyper_settings['num_param_samples'][par] - 1
        # data_name = settings['data_folder']
        # sim_data = []
        # result_data['hyper_param_settings_files'] = []
        params_tmp = []
        for i in range(samples+1):
            if (hyper_settings['param_data_type'][par] == "int"):
                param_value = int( ((range_[1] - range_[0]) * (float(i)/samples)) + range_[0] )
            elif (hyper_settings['param_data_type'][par] == "bool"):
                if ( i == 0):
                    param_value = True
                elif ( i == 1):
                    param_value = False
                else:
                    print("Error to many samples for bool type:")
                    sys.exit()
            else: #float
                delta_ = compute_next_val(range_, i, samples, curve_scheme=hyper_settings['curve_scheme'][par])
                # print ("detla: ", delta_)
                param_value = (delta_) + range_[0]
            params_tmp.append(param_value)
        params_.append(params_tmp)
        
    
    print ("params_: ", params_)
    
    if ( len(params_) > 1 ):
        params_ = list(itertools.product(*params_))
    else:
        params__ = []
        for pars in params_[0]:
            params__.append([pars])
        params_ = params__
    # print ("cross other: ", list(itertools.product(*params_)) )
        
    print ("Cross product of params: ", params_)
    # sys.exit()
    return params_
    

def tuneHyperParameters(simsettingsFileName, hyperSettings=None, saved_fd_model_path=None):
    """
        For some set of parameters the function will sample a number of them
        In order to find a more optimal configuration.
    """
    import os
    
    result_data = {}
    
    file = open(simsettingsFileName)
    settings = json.load(file)
    print ("Settings: " + str(json.dumps(settings, indent=4)))
    file.close()
    file = open(hyperSettings)
    hyper_settings = json.load(file)
    print ("Settings: " + str(json.dumps(settings, indent=4)))
    file.close()
    num_sim_samples = hyper_settings['meta_sim_samples']
    
    ## Check to see if there exists a saved fd model, if so save the path in the hyper settings
    if ( not ( saved_fd_model_path is None )):
        directory= getDataDirectory(settings)
        # file_name_dynamics=directory+"forward_dynamics_"+"_Best_pretrain.pkl" 
        if not os.path.exists(directory):
            hyper_settings['saved_fd_model_path'] = saved_fd_model_path
            
    
    param_settings = get_param_values(hyper_settings)
    result_data['hyper_param_settings_files'] = []
    sim_data = []
    data_name = settings['data_folder']
    for params in param_settings: ## Loop over each setting of parameters
        data_name_tmp = ""
        for par in range(len(params)): ## Assemble the vector of parameters and data folder name
            param_of_interest = hyper_settings['param_to_tune'][par]
            data_name_tmp = data_name_tmp + "/_" + param_of_interest + "_"+ str(params[par]) + "/"
            settings[param_of_interest] = params[par]
        
        settings['data_folder'] = data_name + data_name_tmp
        directory= getBaseDataDirectory(settings)
        if not os.path.exists(directory):
            os.makedirs(directory)
        # file = open(settingsFileName, 'r')
        
        out_file_name=directory+os.path.basename(simsettingsFileName)
        result_data['hyper_param_settings_files'].append(out_file_name)
        print ("Saving settings file with data to: ", out_file_name)
        print ("settings['data_folder']: ", settings['data_folder'])
        out_file = open(out_file_name, 'w')
        out_file.write(json.dumps(settings, indent=4))
        # file.close()
        
        out_file.close()
        sim_data.append((simsettingsFileName, num_sim_samples, copy.deepcopy(settings), hyper_settings['meta_sim_threads'], copy.deepcopy(hyper_settings)))
    
    # p = ProcessingPool(2)
    p = ThreadPool(hyper_settings['tuning_threads'])
    t0 = time.time()
    result = p.map(_trainMetaModel, sim_data)
    t1 = time.time()
    print ("Hyper parameter tuning complete in " + str(datetime.timedelta(seconds=(t1-t0))) + " seconds")
    result_data['sim_time'] = "Meta model training complete in " + str(datetime.timedelta(seconds=(t1-t0))) + " seconds"
    result_data['meta_sim_result'] = result
    result_data['raw_sim_time_in_seconds'] = t1-t0
    result_data['Number_of_simulations_sampled'] = len(param_settings)
    result_data['Number_of_threads_used'] = hyper_settings['tuning_threads'] 
    print (result)
    return result_data
    

if (__name__ == "__main__"):
    """
        python tuneHyperParameters.py <sim_settings_file> <tuning_settings_file>
        Example:
        python tuneHyperParameters.py settings/navGame/PPO_5D.json settings/navGame/PPO_5D_hyper.json 
    """
    import tarfile
    from util.SimulationUtil import addDataToTarBall, addPicturesToTarBall
    from sendEmail import sendEmail
    
    if (len(sys.argv) == 1):
        print("Please incluse sim settings file")
        print("python tuneHyperParameters.py <sim_settings_file> <tuning_settings_file>")
        sys.exit()
    elif (len(sys.argv) == 2):
        print("Please incluse sim settings file")
        print("python tuneHyperParameters.py <sim_settings_file> <tuning_settings_file>")
        sys.exit()
    elif (len(sys.argv) == 3):
        result = tuneHyperParameters(sys.argv[1], sys.argv[2])
        
        hyperSettingsFileName = sys.argv[2] 
        file = open(hyperSettingsFileName)
        hyperSettings_ = json.load(file)
        print ("Settings: " + str(json.dumps(hyperSettings_)))
        file.close()
        
        simsettingsFileName = sys.argv[1]
        file = open(simsettingsFileName)
        simSettings_ = json.load(file)
        print ("Settings: " + str(json.dumps(simSettings_, indent=4)))
        file.close()
        
        root_data_dir = getRootDataDirectory(simSettings_)+"/"
        
        ### Create a tar file of all the sim data
        print ("hyperSettings_['param_to_tune']", hyperSettings_['param_to_tune'])
        print ("hyperSettings_['param_to_tune']", makeNiceName(hyperSettings_['param_to_tune']))
        tarFileName = (root_data_dir + simSettings_['data_folder'] + "/_" + makeNiceName(hyperSettings_['param_to_tune']) +'.tar.gz_') ## gmail doesn't like compressed files....so change the file name ending..
        # tarFileName = (simSettings_['agent_name']+simSettings_['data_folder']+hyperSettings_['param_to_tune']+'.tar.gz')
        dataTar = tarfile.open(tarFileName, mode='w:gz')
        for meta_result in result['meta_sim_result']:
            print (meta_result)
            for simsettings_tmp in meta_result['settings_files']:
                addDataToTarBall(dataTar, simsettings_tmp)
        polt_settings_files = []    
        for hyperSetFile in result['hyper_param_settings_files']:
            print("adding ", hyperSetFile, " to tar file")
            addDataToTarBall(dataTar, simsettings_tmp, fileName=hyperSetFile)
            polt_settings_files.append(hyperSetFile)
        
        figure_file_name = root_data_dir + simSettings_['data_folder'] + "/_" + makeNiceName(hyperSettings_['param_to_tune']) + '_'
        
        print("root_data_dir: ", root_data_dir)
        pictureFileName=None
        try:
            plotMetaDataSimulation(root_data_dir, simSettings_, polt_settings_files, folder=figure_file_name)
            ## Add pictures to tar file
            addPicturesToTarBall(dataTar, simSettings_)
            pictureFileName=figure_file_name + "MBAE_Training_curves.png"
        except Exception as e:
            dataTar.close()
            print("Error plotting data there my not be a DISPLAY available.")
            print("Error: ", e)
        dataTar.close()
        
        ## Send an email so I know this has completed
        ## This prints too much data
        result["meta_sim_result"] = None
        contents_ = json.dumps(hyperSettings_, indent=4, sort_keys=True) + "\n" + json.dumps(result, indent=4, sort_keys=True)
        if ( ('testing' in hyperSettings_ and (hyperSettings_['testing']))):
            print("Not simulating, this is a testing run:")
            testing_ = True
        else:
            testing_ = False 
        sendEmail(subject="Simulation complete: " + result['sim_time'], contents=contents_, hyperSettings=hyperSettings_, simSettings=sys.argv[1], dataFile=tarFileName, testing=testing_, 
                  pictureFile=pictureFileName)
    else:
        print("Please specify arguments properly, ")
        print(sys.argv)
        print("python tuneHyperParameters.py <sim_settings_file> <tuning_settings_file>")
    
        
        