import numpy as np


class CircularBuffer:
    """Circular buffer matching Isaac Lab's implementation exactly."""

    def __init__(self, max_len: int, data_shape: tuple):
        """Initialize the circular buffer.

        Args:
            max_len: The maximum length of the circular buffer (must be >= 1)
            data_shape: Shape of each data entry (e.g., (obs_dim,))
        """
        if max_len < 1:
            raise ValueError(f"Buffer size must be > 0, got {max_len}")

        self._max_len = max_len
        self._data_shape = data_shape
        self._pointer = -1  # -1 means not initialized (same as Isaac Lab)
        self._num_pushes = 0
        self._buffer = None  # initialized on first append (same as Isaac Lab)

    @property
    def max_length(self) -> int:
        return self._max_len

    @property
    def current_length(self) -> int:
        """Current valid length in buffer."""
        return min(self._num_pushes, self._max_len)

    @property
    def buffer(self) -> np.ndarray:
        """Get complete buffer with oldest at index 0, newest at end.
        
        Returns:
            shape: (max_len, *data_shape), ordered oldest -> newest
        """
        if self._buffer is None:
            raise RuntimeError("Buffer is empty. Call append() first.")

        buf = self._buffer.copy()
        # Roll so that oldest is at index 0, newest at end
        buf = np.roll(buf, shift=self._max_len - self._pointer - 1, axis=0)
        return buf

    def reset(self):
        """Reset the circular buffer."""
        self._num_pushes = 0
        # Keep pointer as is, just like Isaac Lab
        if self._buffer is not None:
            self._buffer.fill(0.0)

    def append(self, data: np.ndarray):
        """Append data to the circular buffer.

        Args:
            data: Data to append, shape must match data_shape
        """
        if data.shape != self._data_shape:
            raise ValueError(f"Expected shape {self._data_shape}, got {data.shape}")

        # Initialize buffer on first append (same as Isaac Lab)
        if self._buffer is None:
            self._pointer = -1
            self._buffer = np.zeros((self._max_len,) + self._data_shape, dtype=np.float32)

        # Move pointer to next slot
        self._pointer = (self._pointer + 1) % self._max_len

        # Add new data at pointer position
        self._buffer[self._pointer] = data

        # First push: fill entire buffer with first data (Isaac Lab behavior)
        if self._num_pushes == 0:
            self._buffer[:] = data

        # Increment push count
        self._num_pushes += 1

    def __getitem__(self, key: int) -> np.ndarray:
        """Retrieve data in LIFO fashion (0 = most recent, 1 = second most recent, ...).

        Args:
            key: How many steps back (0 = newest, 1 = one step back, ...)

        Returns:
            Data at that position, shape: data_shape
        """
        if self._num_pushes == 0 or self._buffer is None:
            raise RuntimeError("Buffer is empty. Call append() first.")

        # Clamp key to valid range
        valid_key = min(key, self._num_pushes - 1)

        # Calculate index in circular buffer (pointer points to newest)
        index = (self._pointer - valid_key) % self._max_len

        return self._buffer[index].copy()