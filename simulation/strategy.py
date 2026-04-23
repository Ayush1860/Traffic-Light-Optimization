class MaxPressureStrategy:
    def __init__(self):
        self._last_pressure = 0

    def get_pressure(self, queue_lengths):
        """
        Calculate raw pressure (sum of queue lengths).
        """
        return sum(queue_lengths.values())

    def get_reward(self, queue_lengths):
        """
        Calculate reward based on Max Pressure theory.
        Returns the CHANGE in pressure (positive when pressure decreases).
        """
        current_pressure = sum(queue_lengths.values())
        reward = self._last_pressure - current_pressure
        self._last_pressure = current_pressure
        return reward
