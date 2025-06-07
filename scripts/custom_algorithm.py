"""Example of creating a custom algorithm."""

import numpy as np
from wavevibes import AudioProcessor, Algorithm


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


class SimpleLowpass(Algorithm):
    """Simple one-pole lowpass filter."""
    
    def __init__(self, sample_rate: int, block_size: int, channels: int, cutoff_hz: float = 1000):
        super().__init__(sample_rate, block_size, channels)
        
        # Calculate filter coefficient
        omega = 2 * np.pi * cutoff_hz / sample_rate
        self.alpha = omega / (omega + 1)
        
        # Initialize state
        self.state = np.zeros(channels)
    
    def process(self, block: np.ndarray) -> np.ndarray:
        output = np.zeros_like(block)
        
        for i in range(len(block)):
            # One-pole lowpass: y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
            output[i] = self.alpha * block[i] + (1 - self.alpha) * self.state
            self.state = output[i].copy()
        
        return output
    
    def reset(self):
        """Reset filter state."""
        self.state.fill(0)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python example_custom_algorithm.py input.wav output.wav")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Method 1: Using algorithm factory (recommended)
    # The algorithm will be created with the correct parameters from the input file
    processor = AudioProcessor(
        algorithm_factory=SimpleDelay,
        block_size=512,
        delay_ms=200,
        feedback=0.4,
        mix=0.3
    )
    
    # Process the file
    print(f"Applying delay effect to {input_file}...")
    processor.process(input_file, output_file)
    print(f"Output written to {output_file}")
    
    # Method 2: Using pre-initialized algorithm (if you know the file parameters)
    # delay = SimpleDelay(
    #     sample_rate=44100,
    #     block_size=512,
    #     channels=2,
    #     delay_ms=200,
    #     feedback=0.4,
    #     mix=0.3
    # )
    # processor = AudioProcessor(algorithm=delay, block_size=512)
    # processor.process(input_file, output_file)