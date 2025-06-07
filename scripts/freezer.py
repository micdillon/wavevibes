#!/usr/bin/env python3
"""Example of using WaveVibes for audio synthesis."""

import numpy as np

from wavevibes import Algorithm, AudioProcessor
from wavevibes.io import read_wave


class Freezer(Algorithm):
    """Freezes a section of audio into a wavetable based on zero crossings."""

    def __init__(
        self,
        sample_rate: float,
        block_size: int,
        channels: int,
        filename: str,
        start_loc: float = 0.5,
        n_crossings: int = 2,
    ):
        super().__init__(sample_rate, block_size, channels)
        self.fn = filename
        self.loc = start_loc
        self.n_crossings = n_crossings
        self.phase = 0.0
        self.wav_data, self.wav_sr, self.wav_bd = read_wave(self.fn)
        
        # Extract wavetable from the audio data
        self.wavetable = self._extract_wavetable()
        self.wavetable_size = len(self.wavetable)
        self.phase_increment = self.wavetable_size / self.sample_rate

    def _find_zero_crossings(self, data, start_idx, max_search=10000):
        """Find zero crossing indices in the data starting from start_idx."""
        crossings = []
        prev_sample = data[start_idx]
        
        for i in range(start_idx + 1, min(start_idx + max_search, len(data))):
            curr_sample = data[i]
            # Check for zero crossing (sign change or exact zero)
            if (prev_sample >= 0 and curr_sample < 0) or (prev_sample < 0 and curr_sample >= 0):
                crossings.append(i)
                if len(crossings) >= self.n_crossings:
                    break
            prev_sample = curr_sample
        
        return crossings

    def _extract_wavetable(self):
        """Extract a wavetable from the audio data based on zero crossings."""
        # Use first channel if stereo
        if self.wav_data.ndim > 1:
            data = self.wav_data[:, 0]
        else:
            data = self.wav_data
        
        # Calculate start position in samples
        start_sample = int(self.loc * len(data))
        start_sample = max(0, min(start_sample, len(data) - 1))
        
        # Find zero crossings
        crossings = self._find_zero_crossings(data, start_sample)
        
        if len(crossings) < 2:
            # If not enough crossings found, take a fixed size chunk
            end_sample = min(start_sample + 1024, len(data))
            return data[start_sample:end_sample].copy()
        
        # Extract data between first and last zero crossing
        start_idx = crossings[0]
        end_idx = crossings[-1]
        
        return data[start_idx:end_idx].copy()

    def process(self, block):
        n_samples = block.shape[0]
        output = np.zeros_like(block)
        
        # Generate output using wavetable oscillator
        for i in range(n_samples):
            # Linear interpolation for smoother playback
            idx = int(self.phase)
            frac = self.phase - idx
            
            if idx < self.wavetable_size - 1:
                sample = self.wavetable[idx] * (1 - frac) + self.wavetable[idx + 1] * frac
            else:
                sample = self.wavetable[idx]
            
            # Apply to all channels
            for ch in range(self.channels):
                output[i, ch] = sample
            
            # Update phase with wraparound
            self.phase += self.phase_increment
            if self.phase >= self.wavetable_size:
                self.phase -= self.wavetable_size
        
        return output


def main():
    # Create stereo synthesis
    processor_stereo = AudioProcessor(
        algorithm_factory=lambda sr, bs, ch: Freezer(
            sr, bs, ch, 
            filename="../test_files/bobjames1.wav",
            start_loc=0.2,
            n_crossings=3
        ),
        block_size=512,
    )

    processor_stereo.process(
        input_file=None,
        output_file="synthesized_freeze.wav",
        duration=2.0,
        sample_rate=48000,
        channels=2,
        bit_depth=16,
    )


if __name__ == "__main__":
    main()
