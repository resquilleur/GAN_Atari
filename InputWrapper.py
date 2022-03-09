import gym
import gym.spaces
import numpy as np
import cv2


class InputWrapper(gym.ObservationWrapper):
    """
    Preprocessing on input np array:
    1. resize image into predefined size
    2. move color channel axis to a first place
    """

    def __init__(self, image_size, *args):
        super(InputWrapper, self).__init__(*args)
        assert isinstance(self.observation_space, gym.spaces.Box)
        old_space = self.observation_space
        self.observation_space = gym.spaces.Box(self.observation(old_space.low), self.observation(old_space.high),
                                                dtype=np.float32)
        self.IMAGE_SIZE = image_size

    def observation(self, observation):
        # resize image
        new_obs = cv2.resize(observation, self.IMAGE_SIZE)
        # transform (210, 160, 3) -> (3, 210, 160)
        new_obs = np.moveaxis(new_obs, 2, 0)
        return new_obs.astype(np.float32)
