import time

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
        self.captured_image = None


        # 83x100
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, 60, 50), dtype=np.uint8)

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
        self.captured_image = raw

        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

        resized = cv2.resize(gray, (50, 60))

        channel = np.reshape(resized, (1, 60, 50))

        return channel

    def render(self):
        cv2.imshow("Flappy Bird", self.captured_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()

    def close(self):
        cv2.destroyAllWindows()



if __name__ == "__main__":
    env = FlappyBirdEnv()
    obs = env.reset()
    for i in range(10):
        env.get_observation()