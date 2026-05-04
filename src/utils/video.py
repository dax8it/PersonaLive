from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

try:
    from decord import VideoReader as _DecordVideoReader
except Exception:  # decord has no macOS arm64 wheels; OpenCV fallback is used there.
    _DecordVideoReader = None


def _read_with_cv2(path: str, max_frames: Optional[int] = None) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from video: {path}")
    return np.stack(frames, axis=0)


def read_video_frames(path: str, max_frames: Optional[int] = None, multiple_of: int = 4) -> np.ndarray:
    """Read video frames as RGB numpy array (T, H, W, C).

    Uses decord when available and falls back to OpenCV on macOS arm64, where
    decord==0.6.0 does not publish compatible wheels.
    """
    if _DecordVideoReader is not None:
        reader = _DecordVideoReader(path)
        count = len(reader)
        if max_frames is not None:
            count = min(count, max_frames)
        if multiple_of > 1:
            count = count // multiple_of * multiple_of
        if count <= 0:
            raise RuntimeError(f"No usable frames decoded from video: {path}")
        return reader.get_batch(list(range(count))).asnumpy()

    read_limit = max_frames
    if read_limit is not None and multiple_of > 1:
        read_limit = max(multiple_of, read_limit)
    frames = _read_with_cv2(path, read_limit)
    count = len(frames)
    if max_frames is not None:
        count = min(count, max_frames)
    if multiple_of > 1:
        count = count // multiple_of * multiple_of
    if count <= 0:
        raise RuntimeError(f"Video is shorter than required multiple_of={multiple_of}: {path}")
    return frames[:count]


def read_first_frame(path: str) -> np.ndarray:
    return read_video_frames(path, max_frames=1, multiple_of=1)[0]
