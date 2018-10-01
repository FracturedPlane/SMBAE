

import sys
import subprocess
import os
# sys.path.append('../../TerrainRL/lib')
# print ("os.environ: ", os.environ)
terrainRL_PATH = os.environ['TERRAINRL_PATH']
print ("os.environ[TERRAINRL_PATH]: ", os.environ['TERRAINRL_PATH'])
sys.path.append(terrainRL_PATH+ '/lib')

import simAdapter.terrainRLAdapter