import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)



register(
    id='AntModRightEnv-v0',
    entry_point='gym_antmodright.envs:AntModRightEnv',
    max_episode_steps=1000
)


