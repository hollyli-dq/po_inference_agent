class VirtualClock:
    """
    Virtual clock for Aliyun-Gym.
    Decouples simulation time from wall-clock time.
    """
    def __init__(self):
        self._current_time = 0.0

    def tick(self, seconds: float = 0.1) -> float:
        """Advance virtual time by `seconds`."""
        self._current_time += seconds
        return self._current_time

    def now(self) -> float:
        """Get current virtual time."""
        return self._current_time

    def reset(self) -> None:
        """Reset clock to 0."""
        self._current_time = 0.0
