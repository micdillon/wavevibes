"""Base algorithm interface and example algorithms."""

from abc import ABC, abstractmethod
import numpy as np


class Algorithm(ABC):
    """Base class for audio processing algorithms."""
    
    def __init__(self, sample_rate: int, block_size: int, channels: int):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.channels = channels
    
    @abstractmethod
    def process(self, block: np.ndarray) -> np.ndarray:
        """Process a block of audio data.
        
        Args:
            block: Audio data of shape (block_size, channels) or (block_size,) for mono
            
        Returns:
            Processed audio block with same shape as input
        """
        pass
    
    def reset(self):
        """Reset algorithm state (optional)."""
        pass


class PassThrough(Algorithm):
    """Simple pass-through algorithm for testing."""
    
    def process(self, block: np.ndarray) -> np.ndarray:
        return block.copy()


class Gain(Algorithm):
    """Apply gain to audio signal."""
    
    def __init__(self, sample_rate: int, block_size: int, channels: int, gain_db: float = 0.0):
        super().__init__(sample_rate, block_size, channels)
        self.gain_linear = 10 ** (gain_db / 20.0)
    
    def process(self, block: np.ndarray) -> np.ndarray:
        return block * self.gain_linear


class SimpleDelay(Algorithm):
    """Simple delay effect with feedback."""
    
    def __init__(self, sample_rate: int, block_size: int, channels: int, 
                 delay_ms: float = 250, feedback: float = 0.5, mix: float = 0.5):
        super().__init__(sample_rate, block_size, channels)
        
        # Calculate delay in samples
        self.delay_samples = int(delay_ms * sample_rate / 1000)
        self.feedback = np.clip(feedback, 0, 0.95)  # Prevent runaway feedback
        self.mix = np.clip(mix, 0, 1)
        
        # Initialize delay buffer
        self.delay_buffer = np.zeros((self.delay_samples, channels))
        self.write_index = 0
    
    def process(self, block: np.ndarray) -> np.ndarray:
        output = np.zeros_like(block)
        
        for i in range(len(block)):
            # Read from delay buffer
            delayed = self.delay_buffer[self.write_index]
            
            # Mix dry and wet signals
            output[i] = block[i] * (1 - self.mix) + delayed * self.mix
            
            # Write to delay buffer with feedback
            self.delay_buffer[self.write_index] = block[i] + delayed * self.feedback
            
            # Update write index
            self.write_index = (self.write_index + 1) % self.delay_samples
        
        return output
    
    def reset(self):
        """Reset delay buffer."""
        self.delay_buffer.fill(0)
        self.write_index = 0