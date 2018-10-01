

from trainModel import trainModelParallel
import sys
import json
import copy
from pathos.threading import ThreadPool
from pathos.multiprocessing import ProcessingPool
# from threading import ThreadPool
import time
import datetime

from util.SimulationUtil import getDataDirectory, getBaseDataDirectory, getRootDataDirectory, getAgentName

def _trainMetaModel(input):
    settingsFileName_ = input[0]
    samples_ = input[1]
    settings_ = input[2]
    numThreads_ = input[3]
    if (len(input) > 4 ):
        hyperSettings_ = input[4]
        return trainMetaModel(settingsFileName_, samples=samples_, settings=settings_, numThreads=numThreads_, 
                              hyperSettings=hyperSettings_)
    else:
        return trainMetaModel(settingsFileName_, samples=samples_, settings=settings_, numThreads=numThreads_)
    
    
def trainMetaModel(settingsFileName, samples=10, settings=None, numThreads=1, hyperSettings=None):
    import shutil
    import os
    
    result_data = {}
    result_data['settings_files'] = []
    
    if (settings is None):
        file = open(settingsFileName)
        settings = json.load(file)
        # print ("Settings: " + str(json.dumps(settings)))
        file.close()
    
    print ( "Running ", samples, " simulation(s) over ", numThreads, " Thread(s)")
    settings_original = copy.deepcopy(settings)
    
    directory_= getBaseDataDirectory(settings_original)
    if not os.path.exists(directory_):
        os.makedirs(directory_)
    out_file_name=directory_+"settings.json"
    print ("Saving settings file with data to: ", out_file_name)
    out_file = open(out_file_name, 'w')
    out_file.write(json.dumps(settings_original, indent=4))
    # file.close()
    out_file.close()
        
    sim_settings=[]
    sim_settingFileNames=[]
    sim_data = []
    for i in range(samples):
        settings['data_folder'] = settings_original['data_folder'] + "_" + str(i)
        settings['random_seed'] = int(settings['random_seed']) + ((int(settings['num_available_threads']) + 1) * i)
        ## Change some other settings to reduce memory usage and train faster
        settings['print_level'] = "hyper_train"
        settings['shouldRender'] = False
        settings['visualize_learning'] = False
        settings['saving_update_freq_num_rounds'] = settings_original['saving_update_freq_num_rounds'] * 10
        
        if ( 'expert_policy_files' in settings):
            for j in range(len(settings['expert_policy_files'])):
                settings['expert_policy_files'][j] = settings_original['expert_policy_files'][j] + "/_" + str(i)
                
        result_data['settings_files'].append(copy.deepcopy(settings))
        
        sim_settings.append(copy.deepcopy(settings))
        sim_settingFileNames.append(settingsFileName)
        sim_data.append((settingsFileName,copy.deepcopy(settings)))
        
        ## Create data directory and copy any desired files to these folders .
        if ( not (hyperSettings is None) ):
            # file = open(hyperSettings)
            hyper_settings = hyperSettings
            # print ("Settings: " + str(json.dumps(settings)))
            # file.close()
            directory= getDataDirectory(settings)
            if not os.path.exists(directory):
                os.makedirs(directory)
            if ('saved_model_path' in hyperSettings):
                print ("Copying fd model: ", hyperSettings['saved_model_path'])
                # shutil.copy2(hyperSettings['saved_model_path'], directory+"forward_dynamics_"+"_Best_pretrain.pkl" )
                shutil.copy2(hyperSettings['saved_model_path'], directory+getAgentName()+"_Best.pkl" )
            if ( 'saved_model_folder' in hyperSettings):
                ### Copy models from other metamodel simulation
                ### Purposefully not copying the "Best" model but the last instead
                shutil.copy2(hyperSettings['saved_model_folder']+"/_" + str(i)+'/'+settings['model_type']+'/'+getAgentName()+".pkl", directory+getAgentName()+"_Best.pkl" )
                
        
    # p = ThreadPool(numThreads)
    p = ProcessingPool(numThreads)
    t0 = time.time()
    # print ("hyperSettings: ", hyper_settings)
    if ( (hyperSettings is not None) and ('testing' in hyper_settings and (hyper_settings['testing']))):
        print("Not simulating, this is a testing run:")
    else:
        result = p.map(trainModelParallel, sim_data)
    t1 = time.time()
    print ("Meta model training complete in " + str(datetime.timedelta(seconds=(t1-t0))) + " seconds")
    # print (result)

    result_data['sim_time'] = "Meta model training complete in " + str(datetime.timedelta(seconds=(t1-t0))) + " seconds"
    result_data['raw_sim_time_in_seconds'] = t1-t0
    result_data['Number_of_simulations_sampled'] = samples
    result_data['Number_of_threads_used'] = numThreads
    
    return result_data
    # trainModelParallel(settingsFileName, copy.deepcopy(settings))
        
    
    
def trainMetaModel_(args):
    """
        python trainMetaModel.py <hyper_settings_file> <sim_settings_file> <num_samples> <num_threads> <saved_fd_model_path>
        Example:
        python trainMetaModel.py settings/navGame/PPO_5D.json 10
    """
    from sendEmail import sendEmail
    import json
    import tarfile
    from util.SimulationUtil import addDataToTarBall, addPicturesToTarBall
    from tools.PlotMetadataSimulation import plotMetaDataSimulation
    import os
    
    if (len(args) == 1):
        print("Please incluse sim settings file")
        print("python trainMetaModel.py <sim_settings_file> <hyper_settings_file> <num_samples>")
        sys.exit()
    elif (len(args) == 2):
        print("Please incluse sim settings file")
        print("python trainMetaModel.py <sim_settings_file> <hyper_settings_file> <num_samples>")
        sys.exit()
    elif (len(args) == 3):
        print("Please incluse sim settings file")
        print("python trainMetaModel.py <sim_settings_file> <hyper_settings_file> <num_samples>")
        sys.exit()
    elif ((len(args) == 5) or (len(args) == 6)):
        settingsFileName = args[2] 
        file = open(settingsFileName)
        hyperSettings_ = json.load(file)
        print ("Settings: " + str(json.dumps(hyperSettings_)))
        file.close()
        
        simsettingsFileName = args[1]
        file = open(simsettingsFileName)
        simSettings_ = json.load(file)
        print ("Settings: " + str(json.dumps(simSettings_, indent=4)))
        file.close()
        simSettings_['data_folder'] = simSettings_['data_folder'] + "/"
        
        root_data_dir = getRootDataDirectory(simSettings_)+"/"
        
        if ( len(args) == 6 ):
            hyperSettings_['saved_model_path'] = args[5]
            result = trainMetaModel(args[1], samples=int(args[3]), settings=copy.deepcopy(simSettings_), numThreads=int(args[4]), hyperSettings=hyperSettings_)
        else:
            result = trainMetaModel(args[1], samples=int(args[3]), settings=copy.deepcopy(simSettings_), numThreads=int(args[4]), hyperSettings=hyperSettings_)
        
        directory= getBaseDataDirectory(simSettings_)
        out_file_name=directory+os.path.basename(simsettingsFileName)
        print ("Saving settings file with data to: ", out_file_name)
        out_file = open(out_file_name, 'w')
        out_file.write(json.dumps(simSettings_, indent=4))
        # file.close()
        out_file.close()
        
        ### Create a tar file of all the sim data
        tarFileName = (root_data_dir + simSettings_['data_folder'] + '.tar.gz_') ## gmail doesn't like compressed files....so change the file name ending..
        dataTar = tarfile.open(tarFileName, mode='w:gz')
        for simsettings_tmp in result['settings_files']:
            print ("root_data dir for result: ", getDataDirectory(simsettings_tmp))
            addDataToTarBall(dataTar, simsettings_tmp)
            
        polt_settings_files = []
        polt_settings_files.append(out_file_name)
        # for hyperSetFile in result['hyper_param_settings_files']:
        #     print("adding ", hyperSetFile, " to tar file")
        #     addDataToTarBall(dataTar, simsettings_tmp, fileName=hyperSetFile)
        #     polt_settings_files.append(hyperSetFile)
            
        figure_file_name = root_data_dir + simSettings_['data_folder'] + "/_" + hyperSettings_['param_to_tune'] + '_'
        
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
        result["settings_files"] = None ## Remove extra info
        contents_ = json.dumps(hyperSettings_, indent=4, sort_keys=True) + "\n" + json.dumps(result, indent=4, sort_keys=True)
        sendEmail(subject="Simulation complete: " + result['sim_time'], contents=contents_, hyperSettings=hyperSettings_, simSettings=args[2], dataFile=tarFileName,
                  pictureFile=pictureFileName)    
          
    else:
        print("Please specify arguments properly, ")
        print("args: ", args)
        print("python trainMetaModel.py <sim_settings_file> <hyper_settings_file> <num_samples>")
    

if (__name__ == "__main__"):
        

    trainMetaModel_(sys.argv)
    
    