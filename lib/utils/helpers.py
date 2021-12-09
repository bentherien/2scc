import torch 
import random
import os 
import time
import argparse
import sys
import math

import numpy as np

from multiprocessing import Process


def estimateRemainingTime(trainTime, testTime, currentSteps, totalSteps, currentEpisodes, testStep):
    """Estimates the remaining training time based on imput
    
    Estimates remaining training time by using averages of the 
    each training and test epoch computed. Displays a message 
    indicating averages expected remaining time.

    parameters:
        trainTime -- list of time elapsed for each training epoch
        testTime -- list of time elapsed for each testing epoch
        epochs -- the total number of epochs specified
        currentEpoch -- the current epoch 
        testStep -- episode multiple for obtaining test reward
    """
    remainingSteps = totalSteps - currentSteps

    meanTest = np.mean(testTime)

    trainTime = np.sum(trainTime)
    timePerStep = trainTime / currentSteps

    stepsPerEpisode = currentSteps/currentEpisodes
    projectedRemainingEpisodes = remainingSteps/stepsPerEpisode


    remainingSteps = totalSteps - currentSteps

    remainingTrain = (timePerStep *  remainingSteps) / 60
    remainingTest = (meanTest * (int(projectedRemainingEpisodes / testStep) + 1)) / 60
    remainingTotalMins = remainingTest + remainingTrain

    remainingTotalHours = int(remainingTotalMins/60)
    minutesOver = int(remainingTotalMins - remainingTotalHours*60)

    print("[INFO] ~{}:{} (HH:MM) remaining. Mean train episode duration: {:.2f} s. Mean test duration: {:.2f} s.".format(
        remainingTotalHours,minutesOver, timePerStep*stepsPerEpisode, meanTest
    ))




def setAllSeeds(seed):
    """Helper for setting seeds"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)



def experiments_cli():
    """CLI arguments for experiments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-dr", type=str)
    parser.add_argument("--data-folder", "-df", type=str)
    parser.add_argument("--python", "-p", type=str)

    args = parser.parse_args()

    if args.data_root != None and args.data_folder != None:
        DATA_ARG = "-ddr {} -ddf {}".format(args.data_root,args.data_folder)
    else:
        DATA_ARG = ""

    return sys.executable, DATA_ARG


def experiments_runCommand(cmd):
    """runs one command"""
    print("[Running] {}".format(cmd))
    os.system(cmd)


def experiments_mpCommands(processBatchSize, commands):
    """runs commands in parallel"""
    processes = [Process(target=experiments_runCommand,args=(commands[i],)) for i,cmd in enumerate(commands)]
    processBatches = [processes[i*processBatchSize:(i+1)*processBatchSize] for i in range(math.ceil(len(processes)/processBatchSize))]

    for i,batch in enumerate(processBatches):
        print("Running process batch {}".format(i))
        startTime = time.time()

        for process in batch:
            process.start()
            time.sleep(5)

        for process in batch:
            process.join()

        print("\n\nRunning Took {} seconds".format(time.time() - startTime))
        time.sleep(1)



def parseArgs():
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--env-name', default="HalfCheetah-v2",
                        help='Mujoco Gym environment (default: HalfCheetah-v2)')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--experiment_name', default="Default-Name",
                        help='Name of the experiment passes to wandb')                    
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    parser.add_argument('--load_checkpoint', action="store_true",
                        help='Load from a model checkpoint (default: False)')
    parser.add_argument('--load_filename', default="",
                        help='Then name of a checkpoint to load')
    parser.add_argument('--save_checkpoint', action="store_true",
                        help='Save model checkpoints (default: False)')
    parser.add_argument('--render', action="store_true",
                        help='render the scene (default: False)')
    parser.add_argument('--wandb', action="store_true",
                        help='render the scene (default: False)')
    args = parser.parse_args()

    return args