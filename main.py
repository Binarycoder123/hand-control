from __future__ import annotations

import time

import cv2

from gesture_controller import ControllerConfig, GestureController
from hand_tracker import HandTracker


def draw_overlay(frame, status, fps: float) -> None:
    active_top_left, active_bottom_right = status["active_region"]
    cv2.rectangle(frame, active_top_left, active_bottom_right, (255, 180, 0), 2)

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )

    for line_number, message in enumerate(status["status_lines"], start=1):
        cv2.putText(
            frame,
            message,
            (20, 40 + (line_number * 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )

    cv2.putText(
        frame,
        "q / ESC: quit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (180, 220, 255),
        2,
    )


def main() -> None:
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not capture.isOpened():
        raise RuntimeError("Unable to access webcam. Check that the camera is connected and not in use.")

    try:
        tracker = HandTracker(max_num_hands=2, detection_confidence=0.8, tracking_confidence=0.7)
        controller = GestureController(
            ControllerConfig(
                frame_reduction=100,
                smoothing=6.0,
                click_distance=0.18,
                right_click_distance=0.19,
            )
        )
    except Exception as exc:
        capture.release()
        print(f"Startup failed: {exc}")
        return

    previous_time = time.time()

    try:
        while True:
            success, frame = capture.read()
            if not success:
                continue

            # Mirror the feed so hand motion feels natural on screen.
            frame = cv2.flip(frame, 1)
            hands = tracker.process(frame, draw=True)
            status = controller.process(hands, tracker, frame.shape)

            current_time = time.time()
            fps = 1.0 / max(current_time - previous_time, 1e-6)
            previous_time = current_time

            draw_overlay(frame, status, fps)
            cv2.imshow("Hand Gesture Controlled Windows System", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        controller.shutdown()
        tracker.close()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
