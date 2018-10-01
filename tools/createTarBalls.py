
import tarfile
import sys
sys.path.append("../")

from util.SimulationUtil import addDataToTarBall

if __name__ == '__main__':

    settingsFileName = sys.argv[1] 
    file = open(settingsFileName)
    settings_ = json.load(file)
    print ("Settings: " + str(json.dumps(settings_)))
    file.close()
    print('creating archive')
    out = tarfile.open('tarfile_add.tar.gz', mode='w:gz')
    addDataToTarBall(out, settings_)
    
    
    out.close()
    
    print()
    print('Contents:')
    t = tarfile.open('tarfile_add.tar.gz', mode='r:*')
    for member_info in t.getmembers():
        print(member_info.name)