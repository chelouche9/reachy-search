"""Where question frames come from.

On the robot this is trivially the robot's camera. In development against the
simulator, the "robot camera" is MuJoCo's virtual eye staring at an empty
checkerboard world — useless for showing Reachy a real object — so the source
can be switched to the machine's webcam:

    REACHY_SEARCH_CAMERA=webcam   # always the webcam (the live-demo setting)
    REACHY_SEARCH_CAMERA=robot    # always the robot/sim camera
    REACHY_SEARCH_CAMERA=auto     # robot, falling back to webcam on no frame
                                  # (the default)

The webcam path needs opencv (`uv pip install -e '.[dev]'`) and, on macOS, a
one-time camera permission prompt for the terminal running the app.
"""

import logging
import os

logger = logging.getLogger(__name__)

MODE = os.environ.get("REACHY_SEARCH_CAMERA", "auto").lower()
WEBCAM_INDEX = int(os.environ.get("REACHY_SEARCH_WEBCAM_INDEX", "0"))


class FrameSource:
    def __init__(self, mini):
        self._mini = mini
        self._cap = None
        self._announced = False
        if MODE not in ("robot", "webcam", "auto"):
            logger.warning("Unknown REACHY_SEARCH_CAMERA=%r, using 'auto'", MODE)

    def grab(self) -> bytes | None:
        if MODE == "webcam":
            return self._webcam()
        frame = self._robot()
        if frame is None and MODE != "robot":
            if not self._announced:
                logger.info("Robot camera gave no frame; falling back to webcam %d",
                            WEBCAM_INDEX)
                self._announced = True
            return self._webcam()
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _robot(self) -> bytes | None:
        try:
            return self._mini.media.get_frame_jpeg()
        except Exception:
            logger.warning("Robot camera grab failed", exc_info=True)
            return None

    def _webcam(self) -> bytes | None:
        try:
            import cv2
        except ImportError:
            logger.warning("Webcam requested but opencv is not installed "
                           "(uv pip install -e '.[dev]')")
            return None
        try:
            if self._cap is None:
                self._cap = cv2.VideoCapture(WEBCAM_INDEX)
                if not self._cap.isOpened():
                    logger.warning(
                        "Webcam %d did not open. On macOS, grant camera access "
                        "to your terminal (System Settings > Privacy & Security "
                        "> Camera) and restart the app.", WEBCAM_INDEX)
                    self.close()
                    return None
                # First frames are underexposed while the sensor settles.
                for _ in range(5):
                    self._cap.read()
            # The capture buffers frames; flush so we get *now*, not 2s ago.
            for _ in range(2):
                self._cap.read()
            ok, frame = self._cap.read()
            if not ok or frame is None:
                return None
            ok, encoded = cv2.imencode(".jpg", frame)
            return encoded.tobytes() if ok else None
        except Exception:
            logger.warning("Webcam grab failed", exc_info=True)
            return None
