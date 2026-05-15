"""Thread-safe keyboard state for MuJoCo viewer teleop."""

import threading

# GLFW key codes
GLFW_KEY_W = 87
GLFW_KEY_S = 83
GLFW_KEY_A = 65
GLFW_KEY_D = 68
GLFW_KEY_Q = 81
GLFW_KEY_E = 69
GLFW_KEY_R = 82
GLFW_KEY_P = 80
GLFW_KEY_SPACE = 32


class KeyState:
    """Keyboard state for quadruped teleop.

    Attributes (read via snapshot()):
        vx: Forward velocity command
        vy: Lateral velocity command
        yaw: Yaw rate command
        walking: Whether gait is active
        reset: Whether reset was requested (auto-clears after snapshot)
    """

    def __init__(self, vx_step: float = 0.2, vx_max: float = 2.0, vx_min: float = -1.0, vy_val: float = 0.5):
        self._lock = threading.Lock()
        self._vx_step = vx_step
        self._vx_max = vx_max
        self._vx_min = vx_min
        self._vy_val = vy_val

        self.vx = 0.0
        self.vy = 0.0
        self.yaw = 0.0
        self.walking = False
        self.reset = False

    def on_key(self, key: int):
        """GLFW key callback. Pass to launch_passive(key_callback=...)."""
        with self._lock:
            if key == GLFW_KEY_W:
                self.vx = min(self.vx + self._vx_step, self._vx_max)
                self.walking = True
            elif key == GLFW_KEY_S:
                self.vx = max(self.vx - self._vx_step, self._vx_min)
                self.walking = True
            elif key == GLFW_KEY_A:
                self.yaw = 1.0
                self.walking = True
            elif key == GLFW_KEY_D:
                self.yaw = -1.0
                self.walking = True
            elif key == GLFW_KEY_Q:
                self.vy = self._vy_val
                self.walking = True
            elif key == GLFW_KEY_E:
                self.vy = -self._vy_val
                self.walking = True
            elif key == GLFW_KEY_P:
                self.walking = not self.walking
            elif key == GLFW_KEY_SPACE:
                self.vx = 0.0
                self.vy = 0.0
                self.yaw = 0.0
                self.walking = False
            elif key == GLFW_KEY_R:
                self.reset = True

    def snapshot(self) -> tuple:
        """Read current state and clear one-shot flags.

        Returns:
            (vx, vy, yaw, walking, reset)
        """
        with self._lock:
            s = (self.vx, self.vy, self.yaw, self.walking, self.reset)
            self.reset = False
            return s
