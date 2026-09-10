try:
    import opennsfw2
except (ImportError, Exception):
    opennsfw2 = None

from PIL import Image

from modules.typing import Frame

MAX_PROBABILITY = 0.85


def predict_frame(target_frame: Frame) -> bool:
    if not opennsfw2:
        return False
    try:
        image = Image.fromarray(target_frame)
        image = opennsfw2.preprocess_image(image, opennsfw2.Preprocessing.YAHOO)
        model = opennsfw2.make_open_nsfw_model()
        views = numpy.expand_dims(image, axis=0)
        _, probability = model.predict(views)[0]
        return probability > MAX_PROBABILITY
    except Exception:
        return False


def predict_image(target_path: str) -> bool:
    if not opennsfw2:
        return False
    try:
        return opennsfw2.predict_image(target_path) > MAX_PROBABILITY
    except Exception:
        return False


def predict_video(target_path: str) -> bool:
    if not opennsfw2:
        return False
    try:
        _, probabilities = opennsfw2.predict_video_frames(video_path=target_path, frame_interval=100)
        return any(probability > MAX_PROBABILITY for probability in probabilities)
    except Exception:
        return False
