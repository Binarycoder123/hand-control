from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp


Landmark = Tuple[int, int, int]


@dataclass
class HandState:
    hand_id: int
    handedness: str
    landmarks: List[Landmark]
    normalized_landmarks: List[Tuple[float, float, float]]
    bbox: Tuple[int, int, int, int]
    size: int


class HandTracker:
    def __init__(
        self,
        max_num_hands: int = 2,
        detection_confidence: float = 0.8,
        tracking_confidence: float = 0.7,
        model_path: Optional[str] = None,
    ) -> None:
        self.max_num_hands = max_num_hands
        self.backend = ""
        self._legacy_hands = None
        self._task_landmarker = None
        self._task_image_cls = None
        self._draw_utils = None
        self._draw_styles = None
        self._hand_connections = None
        self._task_timestamp_ms = 0

        if hasattr(mp, "solutions"):
            self._init_legacy_backend(detection_confidence, tracking_confidence)
            return

        self._init_tasks_backend(
            detection_confidence=detection_confidence,
            tracking_confidence=tracking_confidence,
            model_path=model_path,
        )

    def process(self, frame, draw: bool = True) -> List[HandState]:
        if self.backend == "legacy":
            return self._process_legacy(frame, draw=draw)
        if self.backend == "tasks":
            return self._process_tasks(frame, draw=draw)
        return []

    def close(self) -> None:
        if self._legacy_hands is not None:
            self._legacy_hands.close()
        if self._task_landmarker is not None:
            self._task_landmarker.close()

    def fingers_up(self, hand: HandState) -> List[bool]:
        fingers: List[bool] = []

        thumb_tip = hand.landmarks[4]
        thumb_ip = hand.landmarks[3]
        index_mcp = hand.landmarks[5]
        thumb_tip_reach = abs(thumb_tip[0] - index_mcp[0])
        thumb_ip_reach = abs(thumb_ip[0] - index_mcp[0])
        fingers.append(thumb_tip_reach > thumb_ip_reach + 5)

        fingertip_ids = [8, 12, 16, 20]
        pip_ids = [6, 10, 14, 18]
        for tip_id, pip_id in zip(fingertip_ids, pip_ids):
            fingers.append(hand.landmarks[tip_id][1] < hand.landmarks[pip_id][1] - 6)

        return fingers

    def distance(self, hand: HandState, first_id: int, second_id: int) -> float:
        x1, y1, _ = hand.landmarks[first_id]
        x2, y2, _ = hand.landmarks[second_id]
        return hypot(x2 - x1, y2 - y1)

    def normalized_distance(self, hand: HandState, first_id: int, second_id: int) -> float:
        return self.distance(hand, first_id, second_id) / max(hand.size, 1)

    def landmark(self, hand: HandState, landmark_id: int) -> Landmark:
        return hand.landmarks[landmark_id]

    def _init_legacy_backend(self, detection_confidence: float, tracking_confidence: float) -> None:
        self.backend = "legacy"
        self.mp_hands = mp.solutions.hands
        self._draw_utils = mp.solutions.drawing_utils
        self._draw_styles = mp.solutions.drawing_styles
        self._hand_connections = self.mp_hands.HAND_CONNECTIONS
        self._legacy_hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            model_complexity=1,
        )

    def _init_tasks_backend(
        self,
        detection_confidence: float,
        tracking_confidence: float,
        model_path: Optional[str],
    ) -> None:
        resolved_model_path = self._resolve_model_path(model_path)
        if resolved_model_path is None:
            raise RuntimeError(
                "This MediaPipe install uses the newer Tasks API and does not expose "
                "`mediapipe.solutions`. Add a model file at `models/hand_landmarker.task` "
                "or recreate the virtual environment with Python 3.11/3.12 and install a "
                "MediaPipe build that still includes the Solutions API. "
                f"Detected MediaPipe version: {getattr(mp, '__version__', 'unknown')}"
            )

        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import HandLandmarker
        from mediapipe.tasks.python.vision import HandLandmarkerOptions
        from mediapipe.tasks.python.vision import RunningMode
        from mediapipe.tasks.python.vision import drawing_styles
        from mediapipe.tasks.python.vision import drawing_utils
        from mediapipe.tasks.python.vision import hand_landmarker
        from mediapipe.tasks.python.vision.core.image import Image

        self.backend = "tasks"
        self._task_image_cls = Image
        self._draw_utils = drawing_utils
        self._draw_styles = drawing_styles
        self._hand_connections = hand_landmarker.HandLandmarksConnections.HAND_CONNECTIONS
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(resolved_model_path)),
            running_mode=RunningMode.VIDEO,
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._task_landmarker = HandLandmarker.create_from_options(options)

    def _process_legacy(self, frame, draw: bool) -> List[HandState]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._legacy_hands.process(rgb)
        hand_states: List[HandState] = []

        if not results.multi_hand_landmarks or not results.multi_handedness:
            return hand_states

        frame_height, frame_width = frame.shape[:2]
        for hand_id, (landmarks, handedness) in enumerate(
            zip(results.multi_hand_landmarks, results.multi_handedness)
        ):
            label = handedness.classification[0].label
            hand_states.append(
                self._build_hand_state(
                    hand_id=hand_id,
                    handedness=label,
                    landmark_items=landmarks.landmark,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
            )
            if draw:
                self._draw_landmarks(frame, landmarks)

        return hand_states

    def _process_tasks(self, frame, draw: bool) -> List[HandState]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._task_image_cls(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._task_timestamp_ms += 33
        results = self._task_landmarker.detect_for_video(mp_image, self._task_timestamp_ms)
        hand_states: List[HandState] = []

        if not results.hand_landmarks or not results.handedness:
            return hand_states

        frame_height, frame_width = frame.shape[:2]
        for hand_id, (landmarks, handedness) in enumerate(
            zip(results.hand_landmarks, results.handedness)
        ):
            label = handedness[0].category_name if handedness else "Unknown"
            hand_states.append(
                self._build_hand_state(
                    hand_id=hand_id,
                    handedness=label,
                    landmark_items=landmarks,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
            )
            if draw:
                self._draw_landmarks(frame, landmarks)

        return hand_states

    def _draw_landmarks(self, frame, landmarks) -> None:
        self._draw_utils.draw_landmarks(
            frame,
            landmarks,
            self._hand_connections,
            self._draw_styles.get_default_hand_landmarks_style(),
            self._draw_styles.get_default_hand_connections_style(),
        )

    def _build_hand_state(
        self,
        hand_id: int,
        handedness: str,
        landmark_items,
        frame_width: int,
        frame_height: int,
    ) -> HandState:
        points: List[Landmark] = []
        normalized_points: List[Tuple[float, float, float]] = []
        x_values: List[int] = []
        y_values: List[int] = []

        for landmark in landmark_items:
            px = int(landmark.x * frame_width)
            py = int(landmark.y * frame_height)
            pz = int(landmark.z * frame_width)
            points.append((px, py, pz))
            normalized_points.append((landmark.x, landmark.y, landmark.z))
            x_values.append(px)
            y_values.append(py)

        min_x, max_x = min(x_values), max(x_values)
        min_y, max_y = min(y_values), max(y_values)
        width = max_x - min_x
        height = max_y - min_y

        return HandState(
            hand_id=hand_id,
            handedness=handedness,
            landmarks=points,
            normalized_landmarks=normalized_points,
            bbox=(min_x, min_y, width, height),
            size=max(width, height, 1),
        )

    @staticmethod
    def _resolve_model_path(model_path: Optional[str]) -> Optional[Path]:
        candidates = []
        if model_path:
            candidates.append(Path(model_path))
        candidates.append(Path("models") / "hand_landmarker.task")
        candidates.append(Path("hand_landmarker.task"))

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return None
