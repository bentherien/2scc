import os
import sys
sys.path.append(str(os.getcwd()))

from lib.utils.helpers import experiments_cli, experiments_mpCommands

experimentName = __file__.replace("/","-")

PROCESS_BATCH_SIZE = 4

#argument
RUN_FILE = "main.py"
ENV_NAME = "AntModEnv-v0"
POLICY = "Gaussian"
GAMMA = 0.99        #default 0.99
TAU = 0.005            #default 0.005
LR = 0.0003         #default 0.0003
ALPHA = 0.2         #default 0.2 
SEED = 1 
NUM_STEPS = 3000000         #default 1M 
HIDDEN_SIZE = 256       #default 256
UPDATES_PER_STEP = 1        #default 1 
START_STEPS = 10000         #default 10000
TARGET_UPDATE_INTERVAL = 1      #default 1
REPLAY_SIZE = 1000000           #default 1000000
LOAD_FILENAME = "\"\""          #default ""

#Boolean
# EVAL = #True Default
# ENTROPY_TUNING = #False default
SAVE_CHECKPOINT = True
LOAD_CHECKPOINT = False




if __name__ == '__main__':
    PYTHON, DATA_ARG = experiments_cli()

    commands = []

    for ALPHA,SEED in [(0.4,11),(0.5,22),(0.6,33),(0.8,44)]:
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
            args4 += "--save_checkpoint"
        
        if LOAD_CHECKPOINT:
            args4 += "--load_checkpoint"

        command = "{} {} --cuda --wandb {} {} {} {} {}".format(
            PYTHON,RUN_FILE,args1,args2,args3,args4,DATA_ARG)

        commands.append(command)

    
    experiments_mpCommands(
        processBatchSize=PROCESS_BATCH_SIZE,
        commands=commands
    )