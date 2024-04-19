import base64
import time

import numpy as np
from selenium import webdriver
from PIL import Image
import io
from selenium.webdriver.common.by import By


class FlappyBirdCanvasCapture:
    """
    A class for capturing images from the canvas of the Flappy Bird game using Selenium.

    Attributes:
        driver_path (str): The file path of the Chrome WebDriver.
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
        self.canvas_element = None

    def start(self):
        """
        Starts the WebDriver and navigates to the Flappy Bird game URL.
        """
        self.driver = webdriver.Chrome()
        self.driver.get(self.url)
        self.canvas_element = self.driver.find_element(by=By.ID, value='testCanvas')


    def capture_canvas_images(self):
        """
        Captures images from the canvas of the Flappy Bird game and saves them as PNG files.
        """
        if self.driver is None:
            raise Exception("Driver has not been started. Call start() method first.")

        for i in range(self.capture_count):

            canvas_base64 = self.driver.execute_script("return arguments[0].toDataURL('image/png').substring(21);",
                                                       self.canvas_element)
            image = Image.open(io.BytesIO(base64.b64decode(canvas_base64)))

            return np.array(image)

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
        self.driver.execute_script("restart();")


    def stop(self):
        """
        Stops the WebDriver and closes the browser window.
        """
        if self.driver is not None:
            self.driver.quit()


# Example usage:
if __name__ == "__main__":
    # Create a FlappyBirdCanvasCapture object
    canvas_capture = FlappyBirdCanvasCapture()

    # Start the WebDriver and navigate to the game URL
    canvas_capture.start()

    # Check if the bird is dead
    canvas_capture.isDead()

    # Click on the screen to make the bird flap
    canvas_capture.click_screen()

    time.sleep(3)

    # Capture an image from the canvas
    canvas_capture.restart_game()

    time.sleep(3)

    # Stop the WebDriver and close the browser window
    canvas_capture.stop()
