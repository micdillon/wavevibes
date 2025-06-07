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
        audio_data: np.ndarray,
        start_loc: float = 0.5,
        n_crossings: int = 2,
    ):
        super().__init__(sample_rate, block_size, channels)
        self.wav_data = audio_data
        self.loc = start_loc
        self.n_crossings = n_crossings

        # Extract wavetable from the audio data
        self.wavetable = self._extract_wavetable()
        self.wavetable_size = len(self.wavetable)
        print(f"wavetable: {self.wavetable}")
        print(f"wavetable size: {self.wavetable_size}")

        self.phase = 0.0
        self.phase_increment = 1

    def _find_zero_crossings(self, data, start_idx, max_search=10000):
        """Find zero crossing indices in the data starting from start_idx."""
        crossings = []
        prev_sample = data[start_idx]

        for i in range(start_idx + 1, min(start_idx + max_search, len(data))):
            curr_sample = data[i]
            # Check for zero crossing (sign change or exact zero)
            if (prev_sample >= 0 and curr_sample < 0) or (
                prev_sample < 0 and curr_sample >= 0
            ):
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
                sample = (
                    self.wavetable[idx] * (1 - frac) + self.wavetable[idx + 1] * frac
                )
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
    audio_data, samplerate, bit_depth = read_wave("test_files/synth1.wav")
    channels = 2
    block_size = 2048

    freezer = Freezer(
        samplerate, block_size, channels, audio_data, start_loc=0.1, n_crossings=16
    )
    processor_stereo = AudioProcessor(
        algorithm=freezer,
        block_size=block_size,
    )

    processor_stereo.process(
        input_file=None,
        output_file="synthesized_freeze.wav",
        duration=3.0,
        sample_rate=samplerate,
        channels=channels,
        bit_depth=bit_depth,
    )


if __name__ == "__main__":
    main()
