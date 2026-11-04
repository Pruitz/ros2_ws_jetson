from pathlib import Path

import numpy as np
import cv2
import onnxruntime as ort


CLASS_COLORS = {
    0: (0, 0, 0),
    1: (0, 255, 0),
    2: (139, 69, 19),
    3: (0, 0, 255),
}

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def build_onnx_session(model_path: Path):
    if not model_path.is_file():
        raise FileNotFoundError(f"Modello ONNX non trovato: {model_path}")

    available_providers = ort.get_available_providers()
    #requested_providers = ["CPUExecutionProvider"]
    requested_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    session = ort.InferenceSession(str(model_path), providers=requested_providers)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    active_providers = session.get_providers()
    ort_device = ort.get_device()

    return (
        session,
        input_name,
        output_name,
        available_providers,
        requested_providers,
        active_providers,
        ort_device,
    )


def preprocess_frame(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = frame_rgb.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = (img - MEAN) / STD
    return img.astype(np.float32)


def run_inference_on_frame(session, input_name, output_name, frame_bgr):
    input_tensor = preprocess_frame(frame_bgr)
    output = session.run([output_name], {input_name: input_tensor})[0]
    pred_mask = output.argmax(axis=1)[0].astype(np.uint8)
    return pred_mask
