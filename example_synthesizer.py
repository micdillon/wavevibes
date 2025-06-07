#!/usr/bin/env python3
"""Example of using WaveVibes for audio synthesis."""

import numpy as np
from wavevibes import Algorithm, AudioProcessor


class SimpleSineOscillator(Algorithm):
    """A simple sine wave oscillator for synthesis."""

    def __init__(self, sample_rate, block_size, channels, frequency=440.0, amplitude=0.5):
        super().__init__(sample_rate, block_size, channels)
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = 0.0
        self.phase_increment = 2 * np.pi * frequency / sample_rate

    def process(self, block):
        # Generate sine wave regardless of input (synthesis mode sends zeros)
        n_samples = block.shape[0]
        output = np.zeros_like(block)
        
        for ch in range(self.channels):
            for i in range(n_samples):
                output[i, ch] = self.amplitude * np.sin(self.phase)
                self.phase += self.phase_increment
                # Wrap phase to prevent numerical issues
                if self.phase >= 2 * np.pi:
                    self.phase -= 2 * np.pi
        
        return output


def main():
    # Create processor with sine oscillator
    processor = AudioProcessor(
        algorithm_factory=SimpleSineOscillator,
        frequency=440.0,  # A4 note
        amplitude=0.3,
    )
    
    # Synthesize 3 seconds of audio
    processor.process(
        input_file=None,  # No input file - synthesis mode
        output_file="synthesized_sine.wav",
        duration=3.0,
        sample_rate=44100,
        channels=1,
        bit_depth=16,
    )
    
    print("Synthesized sine wave saved to synthesized_sine.wav")
    
    # Create a more complex example with stereo and different frequencies
    class StereoSineOscillator(Algorithm):
        """Stereo sine oscillator with different frequencies per channel."""
        
        def __init__(self, sample_rate, block_size, channels, freq_left=440.0, freq_right=554.37, amplitude=0.3):
            super().__init__(sample_rate, block_size, channels)
            self.freq_left = freq_left
            self.freq_right = freq_right
            self.amplitude = amplitude
            self.phase_left = 0.0
            self.phase_right = 0.0
            self.phase_inc_left = 2 * np.pi * freq_left / sample_rate
            self.phase_inc_right = 2 * np.pi * freq_right / sample_rate
            
        def process(self, block):
            n_samples = block.shape[0]
            output = np.zeros_like(block)
            
            if self.channels >= 2:
                for i in range(n_samples):
                    # Left channel
                    output[i, 0] = self.amplitude * np.sin(self.phase_left)
                    self.phase_left += self.phase_inc_left
                    if self.phase_left >= 2 * np.pi:
                        self.phase_left -= 2 * np.pi
                    
                    # Right channel
                    output[i, 1] = self.amplitude * np.sin(self.phase_right)
                    self.phase_right += self.phase_inc_right
                    if self.phase_right >= 2 * np.pi:
                        self.phase_right -= 2 * np.pi
            
            return output
    
    # Create stereo synthesis
    processor_stereo = AudioProcessor(
        algorithm_factory=StereoSineOscillator,
        freq_left=440.0,   # A4
        freq_right=554.37, # C#5 (major third)
        amplitude=0.25,
    )
    
    processor_stereo.process(
        input_file=None,
        output_file="synthesized_stereo_harmony.wav",
        duration=2.0,
        sample_rate=48000,
        channels=2,
        bit_depth=24,
    )
    
    print("Synthesized stereo harmony saved to synthesized_stereo_harmony.wav")


if __name__ == "__main__":
    main()