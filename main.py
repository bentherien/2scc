"""
This file is a modified version of "main.py" from the following repository
https://github.com/pranz24/pytorch-soft-actor-critic
"""

import datetime
import gym_antmod
import itertools
import gym
import wandb
import time

import matplotlib.pyplot as plt
import numpy as np

from lib.rl.sac import SAC
from lib.rl.replay_memory import ReplayMemory
from lib.utils.visualize import plot
from lib.utils.helpers import estimateRemainingTime, setAllSeeds, parseArgs


def train(args):
    if args.wandb:
        wandb.init(project=args.experiment_name, entity="bentherien")
        wandb.config = args.__dict__

    # Environment
    # env = NormalizedActions(gym.make(args.env_name))
    env = gym.make(args.env_name)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    setAllSeeds(args.seed)

    # Agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args)

    print(args.policy)
    if args.load_filename != "":
        avgRewardList = agent.load_checkpoint(ckpt_path=args.load_filename, evaluate=False)
    else:
        avgRewardList = []


    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)

    # Training Loop
    total_numsteps = 0
    updates = 0


    trainTime = []
    testTime = []

    for i_episode in itertools.count(1):
        t1 = time.time()
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        while not done:
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                    if args.wandb:
                        wandb.log({
                            'loss/critic_1': critic_1_loss,
                            'loss/critic_2': critic_2_loss,
                            'loss/policy': policy_loss,
                            'loss/entropy_loss': ent_loss,
                            'entropy_temprature/alpha': alpha,
                            "Total Steps":total_numsteps,
                            "Episode Number" : i_episode
                        })

            if args.render:
                env.render()
            next_state, reward, done, _ = env.step(action) # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)

            memory.push(state, action, reward, next_state, mask) # Append transition to memory

            state = next_state

        if total_numsteps > args.num_steps:
            break
        
        if args.wandb:
            wandb.log({"Train Reward":episode_reward, "Total Steps":total_numsteps, "Episode Number" : i_episode})

        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
        trainTime.append(time.time()-t1)

        if i_episode % 10 == 0 and args.eval is True:
            t1 = time.time()
            avg_reward = 0.
            episodes = 10
            for _  in range(episodes):
                state = env.reset()
                episode_reward = 0
                done = False
                while not done:
                    action = agent.select_action(state, evaluate=True)

                    next_state, reward, done, _ = env.step(action)
                    episode_reward += reward


                    state = next_state
                avg_reward += episode_reward
            avg_reward /= episodes
            
            if args.wandb:
                wandb.log({"Test Reward":avg_reward, "Total Steps":total_numsteps, "Episode Number" : i_episode})

            avgRewardList.append((avg_reward,total_numsteps))

            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
            print("----------------------------------------")
            testTime.append(time.time()-t1)
            estimateRemainingTime(trainTime=trainTime,testTime=testTime,currentSteps=total_numsteps,totalSteps=args.num_steps,currentEpisodes=i_episode,testStep=1)




        if i_episode % 250 == 0 and args.save_checkpoint is True:
            suffix = "Exp:{}date:{}total_numsteps:{}_lastreward:{:.2f}.pt".format(args.experiment_name,datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), total_numsteps, avg_reward)
            agent.save_checkpoint(env_name=args.env_name, suffix=suffix, testRewardData=avgRewardList)#, ckpt_path="/home/therien/Documents/github/pytorch-soft-actor-critic")

    rewards, steps = np.array([float(x[0]) for x in avgRewardList]), np.array([float(x[1]) for x in avgRewardList])

    assert(len(rewards.shape) == 1)
    assert(len(steps.shape) == 1)

    plot(xvals=steps,yvals=rewards,stderr=None,title="Ant-v3 Higher Level Policy",
         xlab="Episodes",ylab="reward",valLab="Mean reward",color="royalblue",figsize=(17,5)
    )

    if args.wandb:
        wandb.log({"Test Reward Plot:":wandb.Image(plt)})
        plt.savefig("logging/figs/Exp:{}-{}.pdf".format(args.experiment_name,wandb.run.name),bbox_inches="tight")
        wandb.save("logging/figs/Exp:{}-{}.pdf".format(args.experiment_name,wandb.run.name))

        suffix = "final_{}_{}.pt".format(args.experiment_name,wandb.run.name)
        agent.save_checkpoint(env_name=args.env_name, suffix=suffix, testRewardData=avgRewardList)
        wandb.save("logging/checkpoint/{}".format(suffix))

        wandb.log({"avgRewardList":avgRewardList})

    env.close()






















if __name__ == '__main__':
    train(parseArgs())

    exit(0)


