import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)



register(
    id='AntModLeftEnv-v0',
    entry_point='gym_antmodleft.envs:AntModLeftEnv',
    max_episode_steps=1000
)


