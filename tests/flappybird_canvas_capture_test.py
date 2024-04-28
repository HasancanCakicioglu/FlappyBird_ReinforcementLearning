import pytest
import numpy as np

from src.web.cnn.flappybird_canvas_capture import FlappyBirdCanvasCapture


@pytest.fixture(scope="class")
def canvas_capture(request):
    capture = FlappyBirdCanvasCapture()
    capture.start()
    yield capture
    capture.stop()


class TestFlappyBirdCanvasCapture:
    @classmethod
    def setup_class(cls):
        cls.canvas_capture = FlappyBirdCanvasCapture()
        cls.canvas_capture.start()

    @classmethod
    def teardown_class(cls):
        cls.canvas_capture.stop()

    def test_capture_canvas_images(self, canvas_capture):
        image = canvas_capture.capture_canvas_images()
        assert isinstance(image, np.ndarray)
        assert image.shape == (1024, 768, 4)  # Assuming RGBA format for the image

    def test_capture_canvas_images_multiple_times(self, canvas_capture):
        for i in range(5):
            image = canvas_capture.capture_canvas_images()
            assert isinstance(image, np.ndarray)
            assert image.shape == (1024, 768, 4)  # Assuming RGBA format for the image

