import base64
import time

import numpy as np
from selenium import webdriver
from PIL import Image
import io
from selenium.webdriver.common.by import By


class FlappyBirdDataCapture:
    """
    A class for capturing images from the canvas of the Flappy Bird game using Selenium.

    Attributes:
        url (str): The URL of the Flappy Bird game.
        capture_count (int): The number of canvas images to capture.
        wait_time (int): The time to wait for the page to load before capturing images.
        driver (webdriver.Chrome): The WebDriver instance.
    """

    def __init__(self, url="https://flappybird.io/", capture_count=1, wait_time=7):
        """
        Initializes the FlappyBirdCanvasCapture object with default parameters.

        Args:
            url (str, optional): The URL of the Flappy Bird game. Defaults to "https://flappybird.io/".
            capture_count (int, optional): The number of canvas images to capture. Defaults to 20.
            wait_time (int, optional): The time to wait for the page to load before capturing images. Defaults to 7.
        """
        self.url = url
        self.capture_count = capture_count
        self.wait_time = wait_time
        self.driver = None
        self.captured_image = None
        self.passed_pipe_count = 0

        self.bird_y = None

        self.pipe1_x = None
        self.pipe1_lower_y = None
        self.pipe1_upper_y = None

        self.pipe2_x = None
        self.pipe2_lower_y = None
        self.pipe2_upper_y = None

        self.pipe3_x = None
        self.pipe3_lower_y = None
        self.pipe3_upper_y = None

    def start(self):
        """
        Starts the WebDriver and navigates to the Flappy Bird game URL.
        """
        self.driver = webdriver.Chrome()
        self.driver.get(self.url)
        self.canvas_element = self.driver.find_element(by=By.ID, value='testCanvas')

    def capture_datas(self):

        self.bird_y = self.driver.execute_script("return bird.y;")

        pipes_len = self.driver.execute_script("return pipes.children.length;")

        index = self.passed_pipe_count * 2

        if pipes_len >= 6:
            self.pipe1_x = self.driver.execute_script(f"return pipes.children[{index}].x;")
            self.pipe1_lower_y = self.driver.execute_script(f"return pipes.children[{index}].y;")
            self.pipe1_upper_y = self.driver.execute_script(f"return pipes.children[{index+1}].y;")

            self.pipe2_x = self.driver.execute_script(f"return pipes.children[{index+2}].x;")
            self.pipe2_lower_y = self.driver.execute_script(f"return pipes.children[{index+2}].y;")
            self.pipe2_upper_y = self.driver.execute_script(f"return pipes.children[{index+3}].y;")

            self.pipe3_x = self.driver.execute_script(f"return pipes.children[{index+4}].x;")
            self.pipe3_lower_y = self.driver.execute_script(f"return pipes.children[{index+4}].y;")
            self.pipe3_upper_y = self.driver.execute_script(f"return pipes.children[{index+5}].y;")

        if self.pipe1_x is not None and self.pipe1_x < 0:
            self.passed_pipe_count += 1

        #print(f"bird_y: {self.bird_y}, pipe1_x: {self.pipe1_x}, pipe1_lower_y: {self.pipe1_lower_y}, pipe1_upper_y: {self.pipe1_upper_y}, pipe2_x: {self.pipe2_x}, pipe2_lower_y: {self.pipe2_lower_y}, pipe2_upper_y: {self.pipe2_upper_y}, pipe3_x: {self.pipe3_x}, pipe3_lower_y: {self.pipe3_lower_y}, pipe3_upper_y: {self.pipe3_upper_y}")

        return np.array([
            self.bird_y if self.bird_y is not None else 0,
            self.pipe1_x if self.pipe1_x is not None else 0,
            self.pipe1_lower_y if self.pipe1_lower_y is not None else 0,
            self.pipe1_upper_y if self.pipe1_upper_y is not None else 0,
            self.pipe2_x if self.pipe2_x is not None else 0,
            self.pipe2_lower_y if self.pipe2_lower_y is not None else 0,
            self.pipe2_upper_y if self.pipe2_upper_y is not None else 0,
            self.pipe3_x if self.pipe3_x is not None else 0,
            self.pipe3_lower_y if self.pipe3_lower_y is not None else 0,
            self.pipe3_upper_y if self.pipe3_upper_y is not None else 0
        ],dtype=np.float16)

    def isDead(self):
        """
        Checks if the bird is dead in the Flappy Bird game.

        :return:
            bool: True if the bird is dead, False otherwise.
        """

        hit = self.driver.execute_script("if (dead) { return 'True'; } else { return 'False'; }")
        exceeded_high = self.driver.execute_script("if (bird.y < - 50) { return 'True'; } else { return 'False'; }")
        if hit == "True":
            return True
        elif exceeded_high == "True":
            self.driver.execute_script("die();")
            return True
        elif hit == "False":
            return False
        else:
            print("Unknown state")
            return None

    def click_screen(self):
        """
        Clicks on the screen of the Flappy Bird game to make the bird flap.
        """
        self.canvas_element.click()

    def restart_game(self):
        """
        Restarts the game by refreshing the page.
        """

        time.sleep(0.5)
        self.reset_game()
        self.driver.execute_script("restart();")

    def reset_game(self):
        self.passed_pipe_count = 0

        self.bird_y = None

        self.pipe1_x = None
        self.pipe1_lower_y = None
        self.pipe1_upper_y = None

        self.pipe2_x = None
        self.pipe2_lower_y = None
        self.pipe2_upper_y = None

        self.pipe3_x = None
        self.pipe3_lower_y = None
        self.pipe3_upper_y = None


    def stop(self):
        """
        Stops the WebDriver and closes the browser window.
        """
        if self.driver is not None:
            self.driver.quit()


# Example usage:
if __name__ == "__main__":
    # Create a FlappyBirdCanvasCapture object
    canvas_capture = FlappyBirdDataCapture()

    # Start the WebDriver and navigate to the game URL
    canvas_capture.start()

    for i in range(100):
        canvas_capture.capture_datas()
        time.sleep(1)
