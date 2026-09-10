import unittest
from modules.utilities import has_image_extension, is_image, is_video

class TestUtilities(unittest.TestCase):
    def test_has_image_extension(self):
        self.assertTrue(has_image_extension("test.jpg"))
        self.assertTrue(has_image_extension("test.png"))
        self.assertFalse(has_image_extension("test.mp4"))

    def test_is_image(self):
        self.assertFalse(is_image("non_existent_file.jpg"))

    def test_is_video(self):
        self.assertFalse(is_video("non_existent_file.mp4"))

if __name__ == '__main__':
    unittest.main()
