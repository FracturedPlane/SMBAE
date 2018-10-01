#!/bin/bash

###
# These are examples that should work IF YOU REMOVE THE -n from the command
# the -n runs the command as a 'test'

## replace prefix
find ./ -iname "pendulum_agent_*" -exec rename -n 's/pendulum_agent_/agent_/' '{}' \;
## replace algorithm names
find ./ -iname "agent_*" -exec rename -n 's/_A_CACLA|_DPG|_PPO|_algorithm.Distillation.Distillation|_algorithm.A_CACLA.A_CACLA|_A3C|_CACLA_KERAS|_algorithm.MBPG.MBPG|_algorithm.TRPO.TRPO|_algorithm.DPG.DPG|_algorithm.PPO.PPO|_TRPO_Critic|_TRPO//' '{}' \;
