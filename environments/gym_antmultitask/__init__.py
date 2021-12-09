import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)


register(
    id='AntMultitask-v0',
    entry_point='gym_antmultitask.envs:AntMultitask',
    max_episode_steps=1000
)


