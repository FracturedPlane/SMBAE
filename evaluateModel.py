

from model.ModelUtil import *
# import cPickle
import dill
import sys
# from theano.compile.io import Out
sys.setrecursionlimit(50000)
from sim.PendulumEnvState import PendulumEnvState
from sim.PendulumEnv import PendulumEnv
from multiprocessing import Process, Queue
# from pathos.multiprocessing import Pool
import threading
import time
import copy

from actor.ActorInterface import *

import OpenGL
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import numpy as np
np.set_printoptions(threshold=np.nan)

exp=None


fps=30
class SimContainer(object):
    
    def __init__(self, exp, agent, settings, expected_value_viz):
        self._exp = exp
        self._agent = agent
        self._episode=0
        self._settings = settings
        self._grad_sum=0
        self._num_actions=0
        self._expected_value_viz = expected_value_viz
        self._viz_q_values_ = []
        self._action=None
        self._paused=False
        
    def animate(self, callBackVal=-1):
        # print ("Animating: ", callBackVal)
        current_time = glutGet(GLUT_ELAPSED_TIME);
        if (self._paused):
            pass
        else:
            print ("Current sim time: ", current_time)
            """
            glClearColor(0.8, 0.8, 0.9, 0.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_LIGHTING)
            glEnable(GL_NORMALIZE)
            glShadeModel(GL_SMOOTH)
        
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective (45.0, 1.3333, 0.2, 20.0)
        
            glViewport(0, 0, 640, 480)
        
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
        
            glLightfv(GL_LIGHT0,GL_POSITION,[0, 0, 1, 0])
            glLightfv(GL_LIGHT0,GL_DIFFUSE,[1, 1, 1, 1])
            glLightfv(GL_LIGHT0,GL_SPECULAR,[1, 1, 1, 1])
            glEnable(GL_LIGHT0)
        
            glEnable(GL_COLOR_MATERIAL)
            glColor3f(0.8, 0.8, 0.8)
        
            gluLookAt(1.5, 4.0, 3.0, 0.5, 1.0, 0.0, 0.0, 1.0, 0.0)
            
            glutPostRedisplay()
            
            """
            state_ = self._exp.getState()
            # action_ = np.array(self._agent.predict(state_, evaluation_=True), dtype='float64')
            # self._exp.updateAction(action_)
            
            num_substeps = 1
            for i in range(num_substeps):
                # print ("End of Epoch: ", self._exp.getEnvironment().endOfEpoch())
                if (self._exp.getEnvironment().endOfEpoch() and 
                       self._exp.needUpdatedAction()):
                    self._exp.getActor().initEpoch()
                    self._exp.generateValidation(10, self._episode)
                    self._exp.getEnvironment().initEpoch()
                    self._episode += 1
                    print("*******")
                    print("New eposide: ")
                    print("*******")
                    
                """
                simData = self._exp.getEnvironment().getActor().getSimData()
                # print("Average Speed: ", simData.avgSpeed)
                vel_sum = simData.avgSpeed
                torque_sum = simData.avgTorque
                """
                if (self._exp.needUpdatedAction()):
                    state_ = self._exp.getState()
                    # print ("State: ", state_)
                    ## Update value function visualization
                    if ( True  and (self._expected_value_viz is not None)):
                        self._viz_q_values_.append(self._agent.q_value(state_)[0])
                        # self._viz_q_values_.append(0)
                        if (len(self._viz_q_values_)>100):
                             self._viz_q_values_.pop(0)
                        # print ("viz_q_values_: ", viz_q_values_ )
                        # print ("np.zeros(len(viz_q_values_)): ", np.zeros(len(viz_q_values_)))
                        self._expected_value_viz.updateLoss(self._viz_q_values_, np.zeros(len(self._viz_q_values_)))
                        self._expected_value_viz.redraw()
                        # visualizeEvaluation.setInteractiveOff()
                        # visualizeEvaluation.saveVisual(directory+"criticLossGraph")
                        # visualizeEvaluation.setInteractive()
                    """
                    position_root = self._exp.getEnvironment().getActor().getStateEuler()[0:][:3]
                    root_orientation = self._exp.getEnvironment().getActor().getStateEuler()[3:][:3]
                    print("Root position: ", position_root)
                    print("Root orientation: ", root_orientation)
                    """
                    self._action = np.array(self._agent.predict(state_, evaluation_=True), dtype='float64')
                    # self._action = np.array([0.0, 0.0, 0.0, -1.0, 0.0], dtype='float64')
                    # grad_ = self._agent.getPolicy().getGrads(state_)[0]
                    grad_ = [0]
                    self._grad_sum += np.abs(grad_)
                    self._num_actions +=1
                    # print ("Input grad: ", repr(self._grad_sum/self._num_actions))
                    # print ("Input grad: ", str(self._grad_sum/self._num_actions))
                    # print ("Input grad: ", self._grad_sum/self._num_actions)
                    
                    
                    # action_[1] = 1.0
                    print( "New action: ", self._action)
                    self._exp.updateAction(self._action)
                
                if ( self._settings['environment_type'] == 'terrainRLHLCBiped3D' ):
                    self._exp.getActor().updateActor(self._exp, self._action)
                else:
                    self._exp.update()
                
        self._exp.display()
        dur_time = (glutGet(GLUT_ELAPSED_TIME) - current_time)
        next_time = int((1000/fps)) - dur_time
        # print("duration to perform update: ", dur_time, " next time: ", next_time)
        # anim_time = int(gDisplayAnimTime * GetNumTimeSteps() / gPlaybackSpeed);
        # anim_time = np.abs(anim_time);
        # return anim_time;
        next_time = np.max([next_time, 0]);
        glutTimerFunc(int(next_time), self.animate, 0) # 30 fps?
        
    def onKey(self, c, x, y):
        """GLUT keyboard callback."""
    
        global SloMo, Paused
        print ("onKey type: ", type(list(c)[0]))
        print ("onKey type: ", type(c))
        print ("onKey type: ", c)
        print ("onKey type: ", c.decode("utf-8"))
        # set simulation speed
        c = c.decode("utf-8")
        if c >= '0' and c <= '9':
            SloMo = 4 * int(c) + 1
            print ("SLowmo")
        # pause/unpause simulation
        elif c == 'P':
            if ( self._settings["use_parameterized_control"] ):
                self._exp.getActor()._target_lean += 0.025
                print ("Target Height: ", self._exp.getActor()._target_lean)
        elif c == 'p':
            if ( self._settings["use_parameterized_control"] ):
                self._exp.getActor()._target_lean -= 0.025 
                print ("Target Height: ", self._exp.getActor()._target_lean)  
        # quit
        elif c == 'q' or c == 'Q':
            sys.exit(0)
        elif c == 'r':
            print("Resetting Epoch")
            self._exp.getActor().initEpoch()   
            self._exp.getEnvironment().initEpoch()
        elif c == 'M':
            if ( self._settings["use_parameterized_control"] ):
                # self._exp.getActor()._target_vel += 0.1
                self._exp.getActor().setTargetVelocity(self._exp, self._exp.getActor()._target_vel + 0.1)
                print ("Target Velocity: ", self._exp.getActor()._target_vel)
        elif c == 'm':
            if ( self._settings["use_parameterized_control"] ):
                self._exp.getActor().setTargetVelocity(self._exp, self._exp.getActor()._target_vel - 0.1)
                print ("Target Velocity: ", self._exp.getActor()._target_vel)
        elif c == 'H':
            if ( self._settings["use_parameterized_control"] ):
                self._exp.getActor()._target_root_height += 0.02
                print ("Target Height: ", self._exp.getActor()._target_root_height)
        elif c == 'h':
            if ( self._settings["use_parameterized_control"] ):
                self._exp.getActor()._target_root_height -= 0.02
                print ("Target Height: ", self._exp.getActor()._target_root_height)    
        elif c == 'N':
            if ( self._settings["use_parameterized_control"] ):
                self._exp.getActor()._target_hand_pos += 0.02
                print ("_target_hand_pos: ", self._exp.getActor()._target_hand_pos)
        elif c == 'n':
            if ( self._settings["use_parameterized_control"] ):
                self._exp.getActor()._target_hand_pos -= 0.02
                print ("_target_hand_pos: ", self._exp.getActor()._target_hand_pos)
        elif c == ' ':
            self._paused = self._paused != True
            print("Paused: ", self._paused) 
                
        ## ord converts the string to the corresponding integer value for the character...
        self._exp.getEnvironment().onKeyEvent(ord(c), x, y)

def evaluateModelRender(settings_file_name, runLastModel=False, settings=None):

    if ( settings is None):
        settings = getSettings(settings_file_name)
    # settings['shouldRender'] = True
    import os    
    os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device="+settings['training_processor_type']+",floatX="+settings['float_type']
    
    from util.SimulationUtil import validateSettings, createEnvironment, createRLAgent, createActor, getAgentName
    from util.SimulationUtil import getDataDirectory, createForwardDynamicsModel, getAgentName
    from util.ExperienceMemory import ExperienceMemory
    from model.LearningAgent import LearningAgent, LearningWorker
    from RLVisualize import RLVisualize
    from NNVisualize import NNVisualize
    
    model_type= settings["model_type"]
    directory= getDataDirectory(settings)
    rounds = settings["rounds"]
    epochs = settings["epochs"]
    # num_states=settings["num_states"]
    epsilon = settings["epsilon"]
    discount_factor=settings["discount_factor"]
    # max_reward=settings["max_reward"]
    batch_size=settings["batch_size"]
    state_bounds = np.array(settings['state_bounds'])
    action_space_continuous=settings["action_space_continuous"]  
    discrete_actions = np.array(settings['discrete_actions'])
    num_actions= discrete_actions.shape[0]
    reward_bounds=np.array(settings["reward_bounds"])
    action_space_continuous=settings['action_space_continuous']
    if action_space_continuous:
        action_bounds = np.array(settings["action_bounds"], dtype=float)
    
    print ("Sim config file name: " + str(settings["sim_config_file"]))
    
    ### Using a wrapper for the type of actor now
    if action_space_continuous:
        experience = ExperienceMemory(len(state_bounds[0]), len(action_bounds[0]), settings['expereince_length'], continuous_actions=True, settings=settings)
    else:
        experience = ExperienceMemory(len(state_bounds[0]), 1, settings['expereince_length'])
    # actor = ActorInterface(discrete_actions)
    actor = createActor(str(settings['environment_type']),settings, experience)
    masterAgent = LearningAgent(n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                              action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    
    # c = characterSim.Configuration("../data/epsilon0Config.ini")
    if (runLastModel == True):
        file_name=directory+getAgentName()+".pkl"
    else:
        file_name=directory+getAgentName()+"_Best.pkl"
    
    f = open(file_name, 'rb')
    model = dill.load(f)
    f.close()
    
    if (settings['train_forward_dynamics']):
        file_name_dynamics=directory+"forward_dynamics_"+"_Best.pkl"
        # file_name=directory+getAgentName()+".pkl"
        f = open(file_name_dynamics, 'rb')
        forwardDynamicsModel = dill.load(f)
        f.close()
    
    if ( settings["use_transfer_task_network"] ):
        task_directory = getTaskDataDirectory(settings)
        file_name=directory+getAgentName()+"_Best.pkl"
        f = open(file_name, 'rb')
        taskModel = dill.load(f)
        f.close()
        # copy the task part from taskModel to model
        print ("Transferring task portion of model.")
        model.setTaskNetworkParameters(taskModel)

    # this is the process that selects which game to play
    sim_index=0
    if ( 'override_sim_env_id' in settings and (settings['override_sim_env_id'] != False)):
        sim_index = settings['override_sim_env_id']
    exp = createEnvironment(settings["sim_config_file"], settings['environment_type'], settings, render=True, index=sim_index)
    if (settings['train_forward_dynamics']):
        # actor.setForwardDynamicsModel(forwardDynamicsModel)
        forwardDynamicsModel.setActor(actor)
        masterAgent.setForwardDynamics(forwardDynamicsModel)
        # forwardDynamicsModel.setEnvironment(exp)
    # actor.setPolicy(model)
    exp.setActor(actor)
    exp.getActor().init()   
    exp.init()
    exp.generateValidationEnvironmentSample(0)
    expected_value_viz=None
    if (settings['visualize_expected_value'] == True):
        expected_value_viz = NNVisualize(title=str("Expected Value") + " with " + str(settings["model_type"]), settings=settings)
        expected_value_viz.setInteractive()
        expected_value_viz.init()
        criticLosses = []
        
    masterAgent.setSettings(settings)
    masterAgent.setExperience(experience)
    masterAgent.setPolicy(model)
    
    """
    mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error, mean_eval, std_eval = evalModel(actor, exp, masterAgent, discount_factor, anchors=_anchors[:settings['eval_epochs']], 
                                                                                                                        action_space_continuous=action_space_continuous, settings=settings, print_data=True, evaluation=True,
                                                                                                                        visualizeEvaluation=expected_value_viz)
        # simEpoch(exp, model, discount_factor=discount_factor, anchors=_anchors[:settings['eval_epochs']][9], action_space_continuous=True, settings=settings, print_data=True, p=0.0, validation=True)
    """
    """
    workers = []
    input_anchor_queue = Queue(settings['queue_size_limit'])
    output_experience_queue = Queue(settings['queue_size_limit'])
    for process in range(settings['num_available_threads']):
         # this is the process that selects which game to play
        exp = characterSim.Experiment(c)
        if settings['environment_type'] == 'pendulum_env_state':
            print ("Using Environment Type: " + str(settings['environment_type']))
            exp = PendulumEnvState(exp)
        elif settings['environment_type'] == 'pendulum_env':
            print ("Using Environment Type: " + str(settings['environment_type']))
            exp = PendulumEnv(exp)
        else:
            print ("Invalid environment type: " + str(settings['environment_type']))
            sys.exit()
                
        
        exp.getActor().init()   
        exp.init()
        
        w = SimWorker(input_anchor_queue, output_experience_queue, exp, model, discount_factor, action_space_continuous=action_space_continuous, 
                settings=settings, print_data=False, p=0.0, validation=True)
        w.start()
        workers.append(w)
        
    mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error = evalModelParrallel(
        input_anchor_queue, output_experience_queue, discount_factor, anchors=_anchors[:settings['eval_epochs']], action_space_continuous=action_space_continuous, settings=settings)
    
    for w in workers:
        input_anchor_queue.put(None)
       """ 
    # print ("Average Reward: " + str(mean_reward))
    
    exp.getActor().initEpoch()   
    exp.initEpoch()
    fps=30
    if ( settings['environment_type'] == 'terrainRLHLCBiped3D' ): 
        exp._num_updates_since_last_action=1000000
    # state_ = exp.getState()
    # action_ = np.array(masterAgent.predict(state_, evaluation_=True), dtype='float64')
    # exp.updateAction(action_)
    sim = SimContainer(exp, masterAgent, settings, expected_value_viz)
    # sim._grad_sum = np.zeros_like(state_)
    # glutInitWindowPosition(x, y);
    # glutInitWindowSize(width, height);
    # glutCreateWindow("PyODE Ragdoll Simulation")
    # set GLUT callbacks
    glutKeyboardFunc(sim.onKey)
    ## This works because GLUT in C++ uses the same global context (singleton) as the one in python 
    glutTimerFunc(int(1000.0/fps), sim.animate, 0) # 30 fps?
    # glutIdleFunc(animate)
    # enter the GLUT event loop
    glutMainLoop()
    
    
if __name__ == "__main__":
    
    import time
    import datetime
    from util.simOptions import getOptions
    
    options = getOptions(sys.argv)
    options = vars(options)
    print("options: ", options)
    print("options['configFile']: ", options['configFile'])
        
    
    
    file = open(options['configFile'])
    settings = json.load(file)
    file.close()
    
    for option in options:
        if ( not (options[option] is None) ):
            print ("Updateing option: ", option, " = ", options[option])
            settings[option] = options[option]
            if ( options[option] == 'true'):
                settings[option] = True
            elif ( options[option] == 'false'):
                settings[option] = False
        # settings['num_available_threads'] = options['num_available_threads']


    evaluateModelRender(sys.argv[1], runLastModel=True, settings=settings)

