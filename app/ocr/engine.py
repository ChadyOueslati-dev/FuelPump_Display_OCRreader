"""PaddleOCR wrapper. The only module that imports paddle."""

from functools import lru_cache

import numpy as np

from app.ocr.detection import Detection
from app.settings import load_ocr_settings


@lru_cache(maxsize=1)
def _engine():
    # Imported lazily because loading paddle takes several seconds.
    from paddleocr import PaddleOCR

    settings = load_ocr_settings()
    options = {}
    if settings.recognition_model_dir:
        # The recognition model fine-tuned on pump digits (scripts/train).
        options["text_recognition_model_name"] = settings.recognition_model_name
        options["text_recognition_model_dir"] = settings.recognition_model_dir
    if settings.device != "cpu":
        options["device"] = settings.device
    else:
        # oneDNN makes CPU inference about 1.7x faster. It needs paddlepaddle
        # 3.2.x: 3.3.1 crashes with it on ("ConvertPirAttribute2RuntimeAttribute
        # not support"), which is why requirements.txt pins 3.2.2.
        options["enable_mkldnn"] = True

    # The document-oriented extras (page rotation, unwarping, text line
    # rotation) are off: a pump display is a single upright panel.
    return PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        **options,
    )


def detect(image: np.ndarray) -> list[Detection]:
    """Run OCR on one BGR image and return every detected text region."""
    detections = []
    for result in _engine().predict(image):
        for text, score, box in zip(
            result["rec_texts"], result["rec_scores"], result["rec_boxes"]
        ):
            x1, y1, x2, y2 = (int(v) for v in box)
            detections.append(Detection(text, float(score), (x1, y1, x2, y2)))
    return detections
