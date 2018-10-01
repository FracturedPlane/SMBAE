
import sys
import json
import os
import re
from PlotMetadataSimulation import plotMetaDataSimulation
# from util.SimulationUtil import getDataDirectory, getBaseDataDirectory, getRootDataDirectory

"""
    plot_meta_simulation.py <settings_file_name> <path_to_data>
    Example:
    python3 tools/plot_meta_simulation.py ../../../backup/Particle_Sim/algorithm.PPO.PPO/ ../../../backup/Particle_Sim/algorithm.PPO.PPO/Nav_Sphere_SMBAE_10D_Hyper_action_learning_rate_0.025/PPO_SMBAE_10D_Hyper.json

After meta train simulation

	python3 tools/plot_meta_simulation.py terrainRLImitateBiped2D/A_CACLA/ terrainRLImitateBiped2D/A_CACLA/Biped2dFull_Multi_Flat_Incline/Multi_tasking.json

"""

if __name__ == "__main__":
    
    settings = None
    if (len(sys.argv) > 2 ):
        settingsFileName = sys.argv[2]
        settingsFile = open(settingsFileName, 'r')
        settings = json.load(settingsFile)
        settingsFile.close()
        path = None
        # getRootDataDirectory(settings)+"/"
        rlv = plotMetaDataSimulation(path, settings, sys.argv[2:])
        # length = int(sys.argv[2])
        # rlv.setLength(length)
        rlv.show()
    else:
        print ("Please specify arguments properly")
        print ("python plot_meta_simulation.py <path_to_data> <settings_file_name> <settings_file_name> ...")
        sys.exit()
