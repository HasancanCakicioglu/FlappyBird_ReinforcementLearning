import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from matplotlib import pyplot as plt

from src.web.flappybird_canvas_capture import FlappyBirdCanvasCapture


class FlappyBirdEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Initialize the FlappyBirdCanvasCapture object
        self.canvas = FlappyBirdCanvasCapture()
        self.canvas.start()


        # 83x100
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, 120, 100), dtype=np.uint8)

        # 0: flap, 1: do nothing
        self.action_space = spaces.Discrete(2)
    def reset(self,seed=None):
        """
        Resets the environment and returns the initial observation.

        :return:
            observation (np.ndarray): The initial observation.
        """
        if self.canvas.isDead():
            self.canvas.restart_game()
        self.canvas.click_screen()

        return self.get_observation() , {}

    def step(self, action):

        # action 0: flap, action 1: do nothing

        if action == 0:
            self.canvas.click_screen()

        done = self.canvas.isDead()

        new_observation = self.get_observation()

        if done:
            reward = -100
        else:
            reward = 1

        return new_observation, reward, done, False , {}


    def get_observation(self):
        raw = self.canvas.capture_canvas_images()[:,:,:3].astype(np.uint8)

        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

        resized = cv2.resize(gray, (100, 120), interpolation=cv2.INTER_AREA)

        channel = np.reshape(resized, (1, 120, 100))

        return channel

    def render(self):
        return self.render()

    def close(self):
        return self.close()


