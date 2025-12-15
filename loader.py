import numpy as np
import cv2
import json
from math import pi

def load_extrinsics(json_path):
    with open(json_path, 'r') as file:
        extrinsics_json = json.load(file)

    extrinsics = {}
    for k,v in extrinsics_json.items():
        model = np.array(v['matrix_world'])
        model_euler = np.array(v['euler_deg_xyz']) / 180.0 * pi
        view = np.linalg.inv(model)
        frame = int(k) - 1
        extrinsics[frame] = {
            "view": view,
            "model": model,
            "model_euler": model_euler
    }
    return extrinsics

def load_video(video_path):
    """
    Use OpenCV to load a video with filename video_path
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    try:
        frame_nr = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            frame_nr += 1
    except Exception as e:
        raise e
    finally:
        cap.release()
    return frames