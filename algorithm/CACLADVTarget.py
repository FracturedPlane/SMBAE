import theano
from theano import tensor as T
import numpy as np
import lasagne
import sys
import copy
sys.path.append('../')
from model.ModelUtil import *
from algorithm.CACLADV import CACLADV

# For debugging
# theano.config.mode='FAST_COMPILE'
# from DeepCACLA import DeepCACLA

class CACLADVTarget(CACLADV):
    
    def __init__(self, model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):

        super(CACLADVTarget,self).__init__(model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)
        
        # create a small convolutional neural network
        
        # Target network
        self._model_target = copy.deepcopy(model)
        self._modelTarget_target = copy.deepcopy(model)

        
        # Target Networks
        self._q_valsNextState_target = lasagne.layers.get_output(self._model_target.getCriticNetwork(), self._model.getResultStateSymbolicVariable())
        self._q_valsTargetNextState_target = lasagne.layers.get_output(self._modelTarget_target.getCriticNetwork(), self._model.getResultStateSymbolicVariable())
        
        self._q_valsAct_target = lasagne.layers.get_output(self._model_target.getActorNetwork(), self._model.getStateSymbolicVariable())
        self._q_valsActTarget_target = lasagne.layers.get_output(self._modelTarget_target.getActorNetwork(), self._model.getStateSymbolicVariable())
        
        self._q_funcAct_target = self._q_valsAct_target
        self._q_funcActB_target = self._q_valsActTarget_target
        
        target = (self._model.getRewardSymbolicVariable() + (self._discount_factor * self._q_valsNextState_target))
        self._diff = target - self._q_func
        loss = 0.5 * self._diff ** 2
        self._loss = T.mean(loss)
                
        targetTarget = (self._model.getRewardSymbolicVariable() + (self._discount_factor * self._q_valsTargetNextState_target))
        diffTarget = targetTarget - self._q_funcTarget
        lossTarget = 0.5 * diffTarget ** 2 
        self._lossTarget = T.mean(lossTarget)
        
        # TD update
        updates_ = lasagne.updates.rmsprop(T.mean(self._q_func) + (self._regularization_weight * lasagne.regularization.regularize_network_params(
        self._model.getCriticNetwork(), lasagne.regularization.l2)), self._params, 
                    self._learning_rate * -T.mean(self._diff), self._rho, self._rms_epsilon)
        updates_Target = lasagne.updates.rmsprop(T.mean(self._q_funcTarget) + (self._regularization_weight * lasagne.regularization.regularize_network_params(
        self._modelTarget.getCriticNetwork(), lasagne.regularization.l2)), self._paramsTarget, 
                    self._learning_rate * -T.mean(self._diffTarget), self._rho, self._rms_epsilon)
        
        """ # Not sure I need any of this action stuff...
        actDiff = ((self._model.getActionSymbolicVariable() - self._q_valsActA_target)) # Target network does not work well here?
        actLoss = 0.5 * actDiff ** 2 
        self._actLoss = T.sum(actLoss)/float(self._batch_size)
        
        actDiff_Target = ((self._model.getActionSymbolicVariable() - self._q_valsActB_target)) # Target network does not work well here?
        actLoss_Target = 0.5 * actDiff_Target ** 2
        self._actLoss_Target = T.sum(actLoss_Target)/float(self._batch_size)
        
        self._actionUpdates = lasagne.updates.rmsprop(self._actLoss + 
            (self._regularization_weight * lasagne.regularization.regularize_network_params(
                self._model.getActorNetwork(), lasagne.regularization.l2)), self._actionParams, 
                    self._learning_rate , self._rho, self._rms_epsilon)
        self._actionUpdatesTarget = lasagne.updates.rmsprop(self._actLoss_Target + 
          (self._regularization_weight * lasagne.regularization.regularize_network_params(
              self._modelTarget.getActorNetwork(), lasagne.regularization.l2)), self._actionParamsTarget, 
                  self._learning_rate * (-T.sum(actDiff_Target)/float(self._batch_size)), self._rho, self._rms_epsilon)
        """
        
        CACLADVTarget.compile(self)
    
    def compile(self):
        super(CACLADVTarget, self).compile()  
        #### Functions ####
        
    def updateTargetModel(self):
        print ("Updating target Models")
        """
            Target model updates
        """
        all_paramsA = lasagne.layers.helper.get_all_param_values(self._model.getCriticNetwork())
        all_paramsActA = lasagne.layers.helper.get_all_param_values(self._model.getActorNetwork())
        lasagne.layers.helper.set_all_param_values(self._model_target.getCriticNetwork(), all_paramsA)
        lasagne.layers.helper.set_all_param_values(self._model_target.getActorNetwork(), all_paramsActA)
        
        all_paramsA = lasagne.layers.helper.get_all_param_values(self._modelTarget.getCriticNetwork())
        all_paramsActA = lasagne.layers.helper.get_all_param_values(self._modelTarget.getActorNetwork())
        lasagne.layers.helper.set_all_param_values(self._modelTarget_target.getCriticNetwork(), all_paramsA)
        lasagne.layers.helper.set_all_param_values(self._modelTarget_target.getActorNetwork(), all_paramsActA) 
        
    def getNetworkParameters(self):
        params = []
        params.append(lasagne.layers.helper.get_all_param_values(self._model.getCriticNetwork()))
        params.append(lasagne.layers.helper.get_all_param_values(self._model.getActorNetwork()))
        params.append(lasagne.layers.helper.get_all_param_values(self._modelTarget.getCriticNetwork()))
        params.append(lasagne.layers.helper.get_all_param_values(self._modelTarget.getActorNetwork()))
        params.append(lasagne.layers.helper.get_all_param_values(self._model_target.getCriticNetwork()))
        params.append(lasagne.layers.helper.get_all_param_values(self._model_target.getActorNetwork()))
        params.append(lasagne.layers.helper.get_all_param_values(self._modelTarget_target.getCriticNetwork()))
        params.append(lasagne.layers.helper.get_all_param_values(self._modelTarget_target.getActorNetwork()))
        return params
    
    def setNetworkParameters(self, params):
        lasagne.layers.helper.set_all_param_values(self._model.getCriticNetwork(), params[0])
        lasagne.layers.helper.set_all_param_values(self._model.getActorNetwork(), params[1])
        lasagne.layers.helper.set_all_param_values(self._modelTarget.getCriticNetwork(), params[2])
        lasagne.layers.helper.set_all_param_values(self._modelTarget.getActorNetwork(), params[3])
        lasagne.layers.helper.set_all_param_values(self._model_target.getCriticNetwork(), params[4])
        lasagne.layers.helper.set_all_param_values(self._model_target.getActorNetwork(), params[5])
        lasagne.layers.helper.set_all_param_values(self._modelTarget_target.getCriticNetwork(), params[6])
        lasagne.layers.helper.set_all_param_values(self._modelTarget_target.getActorNetwork(), params[7])
   