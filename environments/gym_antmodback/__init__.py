import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)



register(
    id='AntModBackEnv-v0',
    entry_point='gym_antmodback.envs:AntModBackEnv',
    max_episode_steps=1000
)


