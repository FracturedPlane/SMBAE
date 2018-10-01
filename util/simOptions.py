from optparse import OptionParser
import sys

def getOptions(_args=None):
    parser = OptionParser()


    parser.add_option("--load_saved_model",
              action="store", dest="load_saved_model", default=None,
              # type=int,
              metavar="STRING", 
              help="Should the system load a pretrained model [true|false|network_and_scales]")
    
    parser.add_option("--train_forward_dynamics",
              action="store", dest="train_forward_dynamics", default=None,
              type='choice',
              choices=['true', 'false', None],
              metavar="STRING", 
              help="Whether or not to train a forward dynamics model as well [true|false|None]")
    
    parser.add_option("--visualize_expected_value",
              action="store", dest="visualize_expected_value", default=None,
              type='choice',
              choices=['production', 'staging', 'true', 'false', None],
              metavar="STRING", 
              help="Should the system load a pretrained model [true|false|network_and_scales]")
    
    parser.add_option("-v", "--verbose",
                      action="store_true", dest="verbose", default=False,
                      help="print status messages to stdout")
    
    parser.add_option("--dataDir", "--datastoreDirectory", "--dataPath", "--data_folder",
                  action="store", dest="data_folder", default=None,
                  metavar="Directory", help="Specify the directory that files will be stored")
    
    parser.add_option("-p", "--processes", "--availableProcesses", "--num_available_threads",
                      "--p",
              action="store", dest="num_available_threads", default=None,
              type=int,
              metavar="INTEGER", help="The number of processes the SteerStats script can use")
    
    parser.add_option("--bootstrap_samples",
              action="store", dest="bootstrap_samples", default=None,
              type=int,
              metavar="INTEGER", 
              help="Number of initial actions to sample before calculating input/output scaling and starting to train.")
    
    parser.add_option("--eval_epochs",
              action="store", dest="eval_epochs", default=None,
              type=int,
              metavar="INTEGER", 
              help="Number of epoch/episode to evaluate the policy over")
    
    parser.add_option("--epochs",
              action="store", dest="epochs", default=None,
              type=int,
              metavar="INTEGER", 
              help="Number of epochs to perform per round")
    
    parser.add_option("--max_epoch_length",
              action="store", dest="max_epoch_length", default=None,
              type=int,
              metavar="INTEGER", 
              help="That max number of action that can be take before the end of an episode/epoch")
    
    parser.add_option("--print_level",
              action="store", dest="print_level", default=None,
#              type=string,
              metavar="STRING", 
              help="Controls the level of information that is printed to the terminal")
    
    parser.add_option("--plotting_update_freq_num_rounds",
              action="store", dest="plotting_update_freq_num_rounds", default=None,
              type=int,
              metavar="INTEGER", 
              help="Controls the number of simulation rounds to perform before evaluating and re-plotting the policy performance")
    
    parser.add_option("--saving_update_freq_num_rounds",
              action="store", dest="plotting_update_freq_num_rounds", default=None,
              type=int,
              metavar="INTEGER", 
              help="Controllers the number of simulation rounds to perform before saving the policy")
    
    parser.add_option("--sim_config_file",
              action="store", dest="sim_config_file", default=None,
              # type=int,
              metavar="STRING", 
              help="Path to the file the contains the settings for the simulation")
    
    parser.add_option("--frameSize", 
          action="store", dest="frameSize", default=None,
          metavar="IntegerxInteger", help="The pixel width and height, example 640x480")
    
    parser.add_option("--visualize_learning", "--plot", 
          action="store", dest="visualize_learning", default=None,
          type='choice',
          choices=['true', 'false', None],
          metavar="STRING", 
          help="Whether or not to draw/render the simulation")
    
    parser.add_option("--shouldRender", "--render",
          action="store", dest="shouldRender", default=None,
          type='choice',
          choices=['true', 'false', None],
          metavar="STRING", 
          help="TO specify if an openGL window should be created")

    parser.add_option("--config", 
           action="store", metavar="STRING", dest="configFile", default=None,
          help="""The json config file that many of the config settings can be parsed from""")
    
    parser.add_option("--metaConfig", 
           action="store", metavar="STRING", dest="metaConfigFile", default=None,
          help="""The json config file that many of the config settings can be parsed from""")
    
    parser.add_option("--randomSeed", 
           action="store", dest="randomSeed", metavar="INTEGER", default=None,
           help="""randomSeed that will be used for random scenario generation.""")
    
    if _args is None:
        (options, args) = parser.parse_args()
    else:
        (options, args) = parser.parse_args(_args)

    return options
# print getOptions()

