from typing import Any, List
import os
import cv2
import numpy as np
import insightface
import threading

import modules.globals
import modules.processors.frame.core
from modules.core import update_status
from modules.face_analyser import get_one_face, get_many_faces
from modules.typing import Face, Frame
from modules.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'REACTOR.FACE-SWAPPER'


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx'])
    return True


def pre_start() -> bool:
    if not is_image(modules.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False
    elif not get_one_face(cv2.imread(modules.globals.source_path)):
        update_status('No face in source path detected.', NAME)
        return False
    if not is_image(modules.globals.target_path) and not is_video(modules.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_paths = [
                resolve_relative_path('../models/simswap_512.onnx'),
                resolve_relative_path('../models/blendswap_256.onnx'),
                resolve_relative_path('../models/inswapper_128.onnx')
            ]
            chosen_path = next((p for p in model_paths if os.path.exists(p)), model_paths[-1])
            FACE_SWAPPER = insightface.model_zoo.get_model(chosen_path, providers=modules.globals.execution_providers)
    return FACE_SWAPPER


def apply_color_transfer(target_frame: Frame, swapped_frame: Frame) -> Frame:
    """Reinhard color transfer to match swapped face skin tone with original target image."""
    try:
        target_lab = cv2.cvtColor(target_frame, cv2.COLOR_BGR2LAB).astype("float32")
        swapped_lab = cv2.cvtColor(swapped_frame, cv2.COLOR_BGR2LAB).astype("float32")

        (l_mean_t, l_std_t, a_mean_t, a_std_t, b_mean_t, b_std_t) = (
            target_lab[:, :, 0].mean(), target_lab[:, :, 0].std(),
            target_lab[:, :, 1].mean(), target_lab[:, :, 1].std(),
            target_lab[:, :, 2].mean(), target_lab[:, :, 2].std()
        )
        (l_mean_s, l_std_s, a_mean_s, a_std_s, b_mean_s, b_std_s) = (
            swapped_lab[:, :, 0].mean(), swapped_lab[:, :, 0].std(),
            swapped_lab[:, :, 1].mean(), swapped_lab[:, :, 1].std(),
            swapped_lab[:, :, 2].mean(), swapped_lab[:, :, 2].std()
        )

        l = swapped_lab[:, :, 0] - l_mean_s
        a = swapped_lab[:, :, 1] - a_mean_s
        b = swapped_lab[:, :, 2] - b_mean_s

        l = (l * (l_std_t / (l_std_s + 1e-5))) + l_mean_t
        a = (a * (a_std_t / (a_std_s + 1e-5))) + a_mean_t
        b = (b * (b_std_t / (b_std_s + 1e-5))) + b_mean_t

        result_lab = cv2.merge([
            np.clip(l, 0, 255),
            np.clip(a, 0, 255),
            np.clip(b, 0, 255)
        ]).astype("uint8")

        return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    except Exception:
        return swapped_frame


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)


def process_frame(source_face: Face, temp_frame: Frame) -> Frame:
    if modules.globals.many_faces:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                temp_frame = swap_face(source_face, target_face, temp_frame)
    else:
        target_face = get_one_face(temp_frame)
        if target_face:
            temp_frame = swap_face(source_face, target_face, temp_frame)
    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], progress: Any = None) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        try:
            result = process_frame(source_face, temp_frame)
            cv2.imwrite(temp_frame_path, result)
        except Exception as exception:
            print(exception)
            pass
        if progress:
            progress.update(1)


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    target_frame = cv2.imread(target_path)
    result = process_frame(source_face, target_frame)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    modules.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
