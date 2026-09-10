from typing import Any
import insightface

import modules.globals
from modules.typing import Frame

FACE_ANALYSER = None


def get_face_analyser(det_size: tuple = (640, 640)) -> Any:
    global FACE_ANALYSER

    if FACE_ANALYSER is None:
        FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=modules.globals.execution_providers)
        FACE_ANALYSER.prepare(ctx_id=0, det_size=det_size)
    return FACE_ANALYSER


def get_one_face(frame: Frame) -> Any:
    if frame is None:
        return None
    h, w = frame.shape[:2]
    det_size = (1280, 1280) if min(h, w) >= 1080 else (640, 640)
    faces = get_face_analyser(det_size).get(frame)
    try:
        # Return face with largest bounding box area (width * height)
        return max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    except ValueError:
        return None


def get_many_faces(frame: Frame) -> Any:
    try:
        return get_face_analyser().get(frame)
    except IndexError:
        return None


def get_face_by_index(frame: Frame, face_index: int = 0) -> Any:
    """Retrieve target face selectively by sorted left-to-right positional index."""
    faces = get_many_faces(frame)
    if not faces:
        return None
    sorted_faces = sorted(faces, key=lambda x: x.bbox[0])
    if 0 <= face_index < len(sorted_faces):
        return sorted_faces[face_index]
    return None
