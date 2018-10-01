
import numpy as np
import cPickle
import json

def load_data(dataset):
    ''' Loads the datfrom load import mnistaset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''
    import json
    dataFile = open(dataset)
    s = dataFile.read()
    data = json.loads(s)
    dataFile.close()
    tuples = data["Tuples"]
    states = []
    actions = []
    rewards = []
    nextStates = []
    # print tuples
    for tup in tuples:
        states.append(tup["InitialState:"]["Params"])
        actions.append(tup["Action:"]["ID"])
        rewards.append([tup["Reward:"]])
        nextStates.append(tup["ResultState:"]["Params"])
        
    
    # fixed_states = np.amax(np.abs(states), axis=0)
    # fixed_states = np.divide(states, fixed_states)
    # states = fixed_states    
    # fixed_actions = np.amax(np.abs(actions), axis=0)
    # actions = np.divide(actions, fixed_actions)
    # fixed_reward = np.amax(np.abs(rewards), axis=0)
    # rewards = np.divide(rewards, fixed_reward)
    # fixed_nextStates = np.amax(np.abs(nextStates), axis=0)
    # nextStates = np.divide(nextStates, fixed_nextStates)
    print rewards
    
    
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    nextStates = np.array(nextStates)

    train_set = (states,actions,nextStates,rewards)
    
    # train_set_states, train_set_actions, train_set_result_states, train_set_rewards = shared_dataset(train_set)
    
    rval = train_set
    return rval

file_name='best_regressionRL_model.pkl'
def load_model():
    return cPickle.load(open(file_name))    
    
def save_model(model):
    f = open(file_name, 'w')
    cPickle.dump(model, f)
    f.close()
