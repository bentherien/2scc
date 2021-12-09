import os
import sys
sys.path.append(str(os.getcwd()))

from lib.utils.helpers import experiments_cli, experiments_mpCommands

experimentName = __file__.replace("/","-")

PROCESS_BATCH_SIZE = 1

#argument
RUN_FILE = "hlp_main.py"
ENV_NAME = "AntModLeftEnv-v0"
POLICY = "Deterministic"
GAMMA = 0.99        #default 0.99
TAU = 0.005            #default 0.005
LR = 0.0003         #default 0.0003
ALPHA = 0.2         #default 0.2 
SEED = 1 
NUM_STEPS = 1000000         #default 1M 
HIDDEN_SIZE = 256       #default 256
UPDATES_PER_STEP = 1        #default 1 
START_STEPS = 0         #default 10000
TARGET_UPDATE_INTERVAL = 1      #default 1
REPLAY_SIZE = 1000000           #default 1000000
LOAD_FILENAME = "\"checkpoints/sac_checkpoint_AntModEnv-v0_Exp:experiments-lower_level_policy-AntModEnv-v0-DDPG.pydate:2021-12-09_07-45-00total_numsteps:2979609_lastreward:6794.10.pt\""

#"\"logging/checkpoints/sac_checkpoint_AntModEnv-v0_policy:Deterministicdate:2021-12-05_02-08-59total_numsteps:3000813_lastreward:6506.540505362289.pt\""          #default ""

#Boolean
# EVAL = #True Default
# ENTROPY_TUNING = #False default
SAVE_CHECKPOINT = True
LOAD_CHECKPOINT = False
WANDB = True




if __name__ == '__main__':
    PYTHON, DATA_ARG = experiments_cli()

    commands = []

    for SEED in [22]:
        args1 = "--env-name {} --policy {} --experiment_name {} --gamma {} --tau {}".format(
            ENV_NAME,POLICY,experimentName,GAMMA,TAU
        )

        args2 = "--lr {} --alpha {} --seed {} --num_steps {}".format(
            LR,ALPHA,SEED,NUM_STEPS
            )

        args3 = "--hidden_size {} --updates_per_step {} --start_steps {} --target_update_interval {} --replay_size {}".format(
            HIDDEN_SIZE,UPDATES_PER_STEP,START_STEPS,TARGET_UPDATE_INTERVAL,REPLAY_SIZE 
        )

        args4 = " --load_filename {}  {}  {}  {}  {}".format(
            LOAD_FILENAME,"","","",""
        )

        if SAVE_CHECKPOINT:
            args4 += " --save_checkpoint"
        
        if LOAD_CHECKPOINT:
            args4 += " --load_checkpoint"

        if WANDB:
            args4 += " --wandb"

        command = "{} {} --cuda {} {} {} {} {}".format(
            PYTHON,RUN_FILE,args1,args2,args3,args4,DATA_ARG)

        commands.append(command)

    
    experiments_mpCommands(
        processBatchSize=PROCESS_BATCH_SIZE,
        commands=commands
    )