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
        end_loc: float = None,
        n_crossings: int = 2,
        interp_time: float = 1.0,
        min_distance: float = 0.01,
    ):
        super().__init__(sample_rate, block_size, channels)
        self.wav_data = audio_data
        self.start_loc = start_loc
        self.end_loc = end_loc if end_loc is not None else start_loc
        self.current_loc = start_loc
        self.n_crossings = n_crossings
        self.interp_time = interp_time
        self.min_distance = min_distance

        # Wavetable cache: {location: wavetable_data}
        self.wavetable_cache = {}

        # Extract initial wavetable
        self.current_wavetable = self._get_or_extract_wavetable(self.current_loc)
        self.next_wavetable = None

        # Interpolation state
        self.interp_samples = int(interp_time * sample_rate)
        self.interp_counter = 0
        self.is_interpolating = False

        # Location movement
        self.loc_increment = (self.end_loc - self.start_loc) / (
            interp_time * sample_rate
        )

        # Phase tracking for each wavetable
        self.current_phase = 0.0
        self.next_phase = 0.0
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

    def _extract_wavetable(self, location):
        """Extract a wavetable from the audio data based on zero crossings at specific location."""
        # Use first channel if stereo
        if self.wav_data.ndim > 1:
            data = self.wav_data[:, 0]
        else:
            data = self.wav_data

        # Calculate start position in samples
        start_sample = int(location * len(data))
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

        wave_table = data[start_idx:end_idx].copy()

        # Normalize wavetable to prevent amplitude variations
        max_val = np.max(np.abs(wave_table)) * 0.9
        if max_val > 0:
            wave_table = wave_table / max_val

        return wave_table

    def _get_or_extract_wavetable(self, location):
        """Get wavetable from cache or extract if not present."""
        # Round location to avoid floating point precision issues
        cache_key = round(location, 6)

        if cache_key not in self.wavetable_cache:
            self.wavetable_cache[cache_key] = self._extract_wavetable(location)
            print(
                f"Extracted new wavetable at location {location:.3f}, size: {len(self.wavetable_cache[cache_key])}"
            )

        return self.wavetable_cache[cache_key]

    def _hermite_interpolate(self, y0, y1, y2, y3, frac):
        """Hermite interpolation between y1 and y2, using y0 and y3 for slopes."""
        c0 = y1
        c1 = 0.5 * (y2 - y0)
        c2 = y0 - 2.5 * y1 + 2 * y2 - 0.5 * y3
        c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2)
        return ((c3 * frac + c2) * frac + c1) * frac + c0

    def _get_wavetable_sample(self, wavetable, phase):
        """Get interpolated sample from wavetable using Hermite interpolation."""
        wavetable_size = len(wavetable)
        idx = int(phase)
        frac = phase - idx

        if idx < wavetable_size - 1:
            # Get four points for Hermite interpolation
            y0 = wavetable[idx - 1] if idx > 0 else wavetable[-1]
            y1 = wavetable[idx]
            y2 = wavetable[idx + 1]
            y3 = wavetable[idx + 2] if idx < wavetable_size - 2 else wavetable[0]

            return self._hermite_interpolate(y0, y1, y2, y3, frac)
        else:
            return wavetable[idx]

    def process(self, block):
        n_samples = block.shape[0]
        output = np.zeros_like(block)

        # Generate output using wavetable oscillator
        for i in range(n_samples):
            # Check if we need to start interpolating to a new wavetable
            if (
                not self.is_interpolating
                and abs(self.current_loc - self.end_loc) > self.min_distance
            ):
                # Find next location target
                target_loc = (
                    self.current_loc
                    + np.sign(self.end_loc - self.current_loc) * self.min_distance
                )
                target_loc = np.clip(target_loc, 0.0, 1.0)

                # Get the next wavetable
                self.next_wavetable = self._get_or_extract_wavetable(target_loc)
                self.is_interpolating = True
                self.interp_counter = 0
                self.current_loc = target_loc
                # Initialize next phase to match current phase position
                self.next_phase = 0.0

            # Calculate current sample
            current_sample = self._get_wavetable_sample(
                self.current_wavetable, self.current_phase
            )

            # If interpolating, crossfade with next wavetable
            if self.is_interpolating:
                # Calculate interpolation factor (0 to 1)
                interp_factor = self.interp_counter / float(self.interp_samples)

                # Get sample from next wavetable
                next_sample = self._get_wavetable_sample(
                    self.next_wavetable, self.next_phase
                )

                # Crossfade between wavetables
                sample = (
                    current_sample * (1 - interp_factor) + next_sample * interp_factor
                )

                # Update interpolation counter
                self.interp_counter += 1

                # Check if interpolation is complete
                if self.interp_counter >= self.interp_samples:
                    self.current_wavetable = self.next_wavetable
                    self.current_phase = self.next_phase
                    self.next_wavetable = None
                    self.is_interpolating = False
            else:
                sample = current_sample

            # Apply to all channels
            for ch in range(self.channels):
                output[i, ch] = sample

            # Update phases with wraparound
            self.current_phase += self.phase_increment
            if self.current_phase >= len(self.current_wavetable):
                self.current_phase -= len(self.current_wavetable)

            if self.is_interpolating:
                self.next_phase += self.phase_increment
                if self.next_phase >= len(self.next_wavetable):
                    self.next_phase -= len(self.next_wavetable)

        return output


def main():
    audio_data, samplerate, bit_depth = read_wave("test_files/synth1.wav")
    channels = 2
    block_size = 2048

    # Create freezer that moves from 0.1 to 0.9 over 4 seconds
    freezer = Freezer(
        samplerate,
        block_size,
        channels,
        audio_data,
        start_loc=0.2,
        end_loc=0.3,
        n_crossings=16,
        interp_time=1.0,  # 0.5 second crossfade between wavetables
        min_distance=0.001,  # New wavetable every 5% of file
    )
    processor_stereo = AudioProcessor(
        algorithm=freezer,
        block_size=block_size,
    )

    processor_stereo.process(
        input_file=None,
        output_file="synthesized_freeze.wav",
        duration=30.0,  # Longer duration to hear the movement
        sample_rate=samplerate,
        channels=channels,
        bit_depth=bit_depth,
    )


if __name__ == "__main__":
    main()
