from __future__ import annotations

import time
from ctypes import POINTER, cast
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyautogui

try:
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
except ImportError:
    CLSCTX_ALL = None
    AudioUtilities = None
    IAudioEndpointVolume = None

from hand_tracker import HandState, HandTracker


pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0


@dataclass
class ControllerConfig:
    frame_reduction: int = 100
    smoothing: float = 6.0
    click_distance: float = 0.18
    right_click_distance: float = 0.19
    drag_hold_seconds: float = 0.45
    click_cooldown: float = 0.30
    scroll_trigger_pixels: int = 10
    scroll_speed: float = 1.6
    volume_activation_seconds: float = 0.35
    volume_min_distance: float = 0.12
    volume_max_distance: float = 0.65
    shortcut_hold_seconds: float = 0.8
    shortcut_cooldown: float = 1.0


class WindowsVolumeController:
    def __init__(self) -> None:
        self.available = False
        self._volume = None
        self._min_level = 0.0
        self._max_level = 0.0

        if AudioUtilities is None or IAudioEndpointVolume is None or CLSCTX_ALL is None:
            return

        try:
            speakers = AudioUtilities.GetSpeakers()
            interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self._volume = cast(interface, POINTER(IAudioEndpointVolume))
            self._min_level, self._max_level, _ = self._volume.GetVolumeRange()
            self.available = True
        except Exception:
            self.available = False

    def get_ratio(self) -> Optional[float]:
        if not self.available or self._volume is None:
            return None
        current = self._volume.GetMasterVolumeLevel()
        return float((current - self._min_level) / (self._max_level - self._min_level))

    def set_ratio(self, ratio: float) -> None:
        if not self.available or self._volume is None:
            return
        ratio = float(np.clip(ratio, 0.0, 1.0))
        target = self._min_level + (self._max_level - self._min_level) * ratio
        self._volume.SetMasterVolumeLevel(target, None)

    def step(self, delta_ratio: float) -> None:
        current = self.get_ratio()
        if current is None:
            return
        self.set_ratio(current + delta_ratio)


class GestureController:
    def __init__(self, config: Optional[ControllerConfig] = None) -> None:
        self.config = config or ControllerConfig()
        self.screen_width, self.screen_height = pyautogui.size()
        self.volume_controller = WindowsVolumeController()

        self.prev_cursor_x = 0.0
        self.prev_cursor_y = 0.0

        self.left_pinch_start: Optional[float] = None
        self.left_pinch_was_active = False
        self.right_pinch_was_active = False
        self.dragging = False
        self.scroll_anchor_y: Optional[int] = None
        self.volume_pose_start: Optional[float] = None
        self.volume_mode = False
        self.fist_start: Optional[float] = None

        self.last_left_click = 0.0
        self.last_right_click = 0.0
        self.last_shortcut_time = 0.0
        self.last_volume_ratio: Optional[float] = None

    def process(
        self,
        hands: Sequence[HandState],
        tracker: HandTracker,
        frame_shape: Tuple[int, int, int],
    ) -> Dict[str, object]:
        frame_height, frame_width = frame_shape[:2]
        status_lines: List[str] = []

        if not hands:
            self._reset_transient_state(release_mouse=True)
            status_lines.append("No hand detected")
            if not self.volume_controller.available:
                status_lines.append("Volume control unavailable: install pycaw/comtypes")
            return {
                "status_lines": status_lines,
                "active_region": self._active_region(frame_width, frame_height),
            }

        mouse_hand = self._select_mouse_hand(hands)
        shortcut_status = self._handle_shortcuts(mouse_hand, tracker)
        if shortcut_status:
            status_lines.append(shortcut_status)

        if self._is_open_palm(tracker.fingers_up(mouse_hand)):
            self._reset_pointer_modes(release_mouse=True)
            status_lines.append("Open palm: cursor paused")
        else:
            # Volume is intentionally mode-based so a brief pinch still behaves like a click.
            volume_status = self._handle_volume(mouse_hand, tracker)

            if volume_status:
                status_lines.append(volume_status)
                self._reset_pointer_modes(release_mouse=True)
            else:
                pointer_status = self._handle_pointer(mouse_hand, tracker, frame_width, frame_height)
                if pointer_status:
                    status_lines.extend(pointer_status)

        if self.dragging:
            status_lines.append("Dragging")
        if not self.volume_controller.available:
            status_lines.append("Volume control unavailable: install pycaw/comtypes")

        return {
            "status_lines": status_lines,
            "active_region": self._active_region(frame_width, frame_height),
        }

    def shutdown(self) -> None:
        self._reset_transient_state(release_mouse=True)

    def _handle_pointer(
        self,
        hand: HandState,
        tracker: HandTracker,
        frame_width: int,
        frame_height: int,
    ) -> List[str]:
        fingers = tracker.fingers_up(hand)
        status_lines: List[str] = []
        index_tip = tracker.landmark(hand, 8)
        left_pinched = tracker.normalized_distance(hand, 4, 8) < self.config.click_distance
        right_pinched = tracker.normalized_distance(hand, 4, 12) < self.config.right_click_distance

        if self._is_scroll_pose(fingers):
            scroll_status = self._handle_scroll(index_tip[1], tracker.landmark(hand, 12)[1])
            if scroll_status:
                status_lines.append(scroll_status)
        else:
            self.scroll_anchor_y = None

        if self._can_move_cursor(fingers) or self.dragging:
            self._move_cursor(index_tip[0], index_tip[1], frame_width, frame_height)
            status_lines.append("Cursor control active")

        current_time = time.time()

        if self._can_left_click(fingers):
            if left_pinched:
                if self.left_pinch_start is None:
                    self.left_pinch_start = current_time
                elif (
                    not self.dragging
                    and current_time - self.left_pinch_start >= self.config.drag_hold_seconds
                ):
                    pyautogui.mouseDown()
                    self.dragging = True
                    status_lines.append("Drag started")
            else:
                # A quick pinch-release becomes a click, while a longer hold becomes drag.
                if (
                    self.left_pinch_start is not None
                    and not self.dragging
                    and current_time - self.last_left_click >= self.config.click_cooldown
                ):
                    pyautogui.click()
                    self.last_left_click = current_time
                    status_lines.append("Left click")
                self.left_pinch_start = None
        else:
            self.left_pinch_start = None

        if self.dragging and not left_pinched:
            pyautogui.mouseUp()
            self.dragging = False
            status_lines.append("Drop")

        if self._can_right_click(fingers):
            if (
                right_pinched
                and not self.right_pinch_was_active
                and current_time - self.last_right_click >= self.config.click_cooldown
            ):
                pyautogui.click(button="right")
                self.last_right_click = current_time
                status_lines.append("Right click")

        self.left_pinch_was_active = left_pinched
        self.right_pinch_was_active = right_pinched
        return status_lines

    def _handle_scroll(self, index_y: int, middle_y: int) -> Optional[str]:
        center_y = int((index_y + middle_y) / 2)
        if self.scroll_anchor_y is None:
            self.scroll_anchor_y = center_y
            return "Scroll mode"

        delta = self.scroll_anchor_y - center_y
        self.scroll_anchor_y = center_y

        if abs(delta) < self.config.scroll_trigger_pixels:
            return "Scroll mode"

        pyautogui.scroll(int(delta * self.config.scroll_speed))
        return "Scrolling"

    def _handle_volume(self, hand: HandState, tracker: HandTracker) -> Optional[str]:
        if not self.volume_controller.available:
            self.volume_pose_start = None
            self.volume_mode = False
            return None

        fingers = tracker.fingers_up(hand)
        volume_pose = self._is_volume_pose(fingers)
        current_time = time.time()

        if volume_pose and not self.volume_mode:
            self.volume_pose_start = self.volume_pose_start or current_time
            if current_time - self.volume_pose_start >= self.config.volume_activation_seconds:
                self.volume_mode = True
        elif not volume_pose:
            self.volume_pose_start = None
            self.volume_mode = False

        if not self.volume_mode:
            return None

        distance = tracker.normalized_distance(hand, 4, 8)
        ratio = np.interp(
            distance,
            [self.config.volume_min_distance, self.config.volume_max_distance],
            [0.0, 1.0],
        )
        self.volume_controller.set_ratio(float(ratio))
        self.last_volume_ratio = float(np.clip(ratio, 0.0, 1.0))
        return f"Volume: {int(self.last_volume_ratio * 100)}%"

    def _handle_shortcuts(self, hand: HandState, tracker: HandTracker) -> Optional[str]:
        fingers = tracker.fingers_up(hand)
        current_time = time.time()

        if self._is_fist(fingers):
            self.fist_start = self.fist_start or current_time
            if (
                current_time - self.fist_start >= self.config.shortcut_hold_seconds
                and current_time - self.last_shortcut_time >= self.config.shortcut_cooldown
            ):
                pyautogui.press("playpause")
                self.last_shortcut_time = current_time
                self.fist_start = current_time
                return "Media play/pause"
        else:
            self.fist_start = None

        if (
            self._is_thumbs_up(fingers)
            and self.volume_controller.available
            and current_time - self.last_shortcut_time >= self.config.shortcut_cooldown
        ):
            self.volume_controller.step(0.05)
            self.last_shortcut_time = current_time
            return "Thumbs up: volume up"

        return None

    def _move_cursor(self, x: int, y: int, frame_width: int, frame_height: int) -> None:
        frame_margin = self._effective_margin(frame_width, frame_height)
        mapped_x = np.interp(
            x,
            [frame_margin, max(frame_margin + 1, frame_width - frame_margin)],
            [0, self.screen_width],
        )
        mapped_y = np.interp(
            y,
            [frame_margin, max(frame_margin + 1, frame_height - frame_margin)],
            [0, self.screen_height],
        )

        smooth_x = self.prev_cursor_x + (mapped_x - self.prev_cursor_x) / self.config.smoothing
        smooth_y = self.prev_cursor_y + (mapped_y - self.prev_cursor_y) / self.config.smoothing
        pyautogui.moveTo(int(smooth_x), int(smooth_y))

        self.prev_cursor_x = smooth_x
        self.prev_cursor_y = smooth_y

    def _reset_pointer_modes(self, release_mouse: bool) -> None:
        self.left_pinch_start = None
        self.left_pinch_was_active = False
        self.right_pinch_was_active = False
        self.scroll_anchor_y = None
        if release_mouse and self.dragging:
            pyautogui.mouseUp()
        self.dragging = False

    def _reset_transient_state(self, release_mouse: bool) -> None:
        self._reset_pointer_modes(release_mouse=release_mouse)
        self.volume_pose_start = None
        self.volume_mode = False
        self.fist_start = None

    def _active_region(self, frame_width: int, frame_height: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        margin = self._effective_margin(frame_width, frame_height)
        return (margin, margin), (frame_width - margin, frame_height - margin)

    def _effective_margin(self, frame_width: int, frame_height: int) -> int:
        return max(10, min(self.config.frame_reduction, frame_width // 3, frame_height // 3))

    @staticmethod
    def _select_mouse_hand(hands: Sequence[HandState]) -> HandState:
        right_hands = [hand for hand in hands if hand.handedness.lower() == "right"]
        return right_hands[0] if right_hands else hands[0]

    @staticmethod
    def _can_move_cursor(fingers: Sequence[bool]) -> bool:
        return fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]

    @staticmethod
    def _can_left_click(fingers: Sequence[bool]) -> bool:
        return fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]

    @staticmethod
    def _can_right_click(fingers: Sequence[bool]) -> bool:
        return fingers[2] and not fingers[1] and not fingers[3] and not fingers[4]

    @staticmethod
    def _is_scroll_pose(fingers: Sequence[bool]) -> bool:
        return fingers[1] and fingers[2] and not fingers[3] and not fingers[4]

    @staticmethod
    def _is_volume_pose(fingers: Sequence[bool]) -> bool:
        return fingers[0] and fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]

    @staticmethod
    def _is_open_palm(fingers: Sequence[bool]) -> bool:
        return sum(bool(finger) for finger in fingers) >= 4

    @staticmethod
    def _is_fist(fingers: Sequence[bool]) -> bool:
        return sum(bool(finger) for finger in fingers) == 0

    @staticmethod
    def _is_thumbs_up(fingers: Sequence[bool]) -> bool:
        return fingers[0] and not fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]
