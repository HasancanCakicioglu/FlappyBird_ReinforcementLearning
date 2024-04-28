import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from src.web.mlp.flappybird_data_capture import FlappyBirdDataCapture


class FlappyBirdMlpEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Initialize the FlappyBirdgameCapture object
        self.game = FlappyBirdDataCapture()
        self.game.start()

        # 83x100
        self.observation_space = spaces.Box(low=-200, high=2000, shape=(10,), dtype=np.float16)

        # 0: flap, 1: do nothing
        self.action_space = spaces.Discrete(2)
    def reset(self,seed=None):
        """
        Resets the environment and returns the initial observation.

        :return:
            observation (np.ndarray): The initial observation.
        """
        if self.game.isDead():
            self.game.restart_game()
        self.game.click_screen()

        return self.get_observation() , {}

    def step(self, action):

        # action 0: flap, action 1: do nothing

        if action == 0:
            self.game.click_screen()

        done = self.game.isDead()

        new_observation = self.get_observation()

        if done:
            reward = -100
        else:
            reward = 1
        return new_observation, reward, done, False , {}

    def get_observation(self):
        return self.game.capture_datas()

    def render(self):
        pass

    def close(self):
        cv2.destroyAllWindows()



if __name__ == "__main__":
    env = FlappyBirdMlpEnv()
    obs = env.reset()
    for i in range(10):
        env.get_observation()