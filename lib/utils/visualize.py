import torch
import os

import matplotlib.pyplot as plt
import numpy as np



def plot(xvals,yvals,stderr=None,title="default title",
         xlab="default",ylab="default",valLab="default",
         color="blue",figsize=(7,7),save=False):

    f2,ax = plt.subplots(figsize=figsize)
    plt.title(title,fontsize='x-large')
    plt.xlabel(xlab,fontsize='xx-large')
    plt.ylabel(ylab,fontsize='xx-large')
    plt.plot([x for x in xvals],yvals,label=valLab,color=color,zorder=2)
    if str(type(stderr)) != "<class 'NoneType'>":
        plt.fill_between(range(len(yvals)), np.array(yvals)-np.array(stderr), np.array(yvals)+np.array(stderr), 
                         color="lightsteelblue", alpha=0.3,label="STD Error",zorder=1)

    plt.legend(fontsize='xx-large',loc=2)
    ax.grid(linestyle='--',zorder=0)
    if save:
        plt.savefig('logging/figs/{}.pdf'.format(title.replace(" ","_")),bbox_inches="tight")


def getRewards(filename):   
    filename = os.path.join("logging","checkpoints",filename)
    rewards = torch.load(filename)["testRewardData"]
    return np.array([float(x[0]) for x in rewards]), np.array([float(x[1]) for x in rewards])

def getRewardsAVG(fileList):
    """takes the average over all the runs contained in the file List"""
    rewardMat = None
    stepsMat = None
    for x in range(fileList):
        rewards, steps = getRewards(x)
        if x == 0:
            rewardMat = np.expand_dims(rewards,axis=1)
            stepsMat =  np.expand_dims(steps,axis=1)
        else:
            rewardMat = np.concatenate([rewardMat,np.expand_dims(rewards,axis=1)],axis=1)
            stepsMat = np.concatenate([stepsMat,np.expand_dims(steps,axis=1)],axis=1)

    mean = np.mean(rewardMat,axis=1)
    stderr = np.std(rewardMat, axis=1)/np.sqrt(len(fileList))
    return mean, stderr, steps


if __name__=="__main__":

    file = "sac_checkpoint_AntModEnv-v0_policy:Deterministicdate:2021-12-05_02-08-59total_numsteps:3000813_lastreward:6506.540505362289.pt" #trained over a lower level deterministic policy for 2766541 steps
    fileList =[]

    if fileList != []:
        mean, stderr, steps = getRewardsAVG(fileList)

        assert(len(mean.shape) == 1)
        assert(len(stderr.shape) == 1)
        assert(len(steps.shape) == 1)

        plot(xvals=steps,yvals=mean,stderr=stderr,title="Ant-v3 Higher Level Policy",
        xlab="Episodes",ylab="reward",valLab="Mean reward",color="royalblue",figsize=(17,5))
    else:
        rewards, steps = getRewards(file)

        assert(len(rewards.shape) == 1)
        assert(len(steps.shape) == 1)

        plot(xvals=steps,yvals=rewards,stderr=None,title="AntModEnv-v0 LoweLevelPolicy",
        xlab="Episodes",ylab="reward",valLab="Mean reward",color="royalblue",figsize=(17,5),save=True)
 