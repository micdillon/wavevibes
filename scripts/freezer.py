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
        normalize: bool = True,
    ):
        super().__init__(sample_rate, block_size, channels)
        self.wav_data = audio_data
        self.start_loc = start_loc
        self.end_loc = end_loc if end_loc is not None else start_loc
        self.current_loc = start_loc
        self.n_crossings = n_crossings
        self.interp_time = interp_time
        self.min_distance = min_distance
        self.normalize = normalize

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

    def _find_synchronized_crossings(
        self, data, start_idx, max_search=10000, sync_window=5
    ):
        """Find zero crossings that occur close together in both channels."""
        if data.ndim == 1:
            # Mono - just return regular crossings
            return self._find_zero_crossings(data, start_idx, max_search), None

        # Find crossings in both channels
        left_crossings = self._find_zero_crossings(data[:, 0], start_idx, max_search)
        right_crossings = self._find_zero_crossings(data[:, 1], start_idx, max_search)

        # Find synchronized crossings (within sync_window samples of each other)
        sync_left = []
        sync_right = []

        for lc in left_crossings:
            # Find closest right crossing
            if right_crossings:
                distances = [abs(lc - rc) for rc in right_crossings]
                min_dist_idx = np.argmin(distances)

                if distances[min_dist_idx] <= sync_window:
                    sync_left.append(lc)
                    sync_right.append(right_crossings[min_dist_idx])

                    if len(sync_left) >= self.n_crossings:
                        break

        # If not enough synchronized crossings, fall back to independent crossings
        if len(sync_left) < 2:
            return (
                left_crossings[: self.n_crossings],
                right_crossings[: self.n_crossings],
            )

        return sync_left, sync_right

    def _extract_wavetable(self, location):
        """Extract a stereo wavetable from the audio data based on synchronized zero crossings."""
        # Ensure we have stereo data to work with
        if self.wav_data.ndim == 1:
            # Duplicate mono to stereo
            data = np.column_stack((self.wav_data, self.wav_data))
        else:
            data = self.wav_data

        # Calculate start position in samples
        start_sample = int(location * len(data))
        start_sample = max(0, min(start_sample, len(data) - 1))

        # Find synchronized zero crossings
        left_crossings, right_crossings = self._find_synchronized_crossings(
            data, start_sample
        )

        # Handle cases where not enough crossings found
        if left_crossings is None or len(left_crossings) < 2:
            # Fallback to fixed size chunk
            print("Falling back to fixed size chunk !!!")
            end_sample = min(start_sample + 1024, len(data))
            left_wave = data[start_sample:end_sample, 0].copy()
            right_wave = (
                data[start_sample:end_sample, 1].copy()
                if data.shape[1] > 1
                else left_wave.copy()
            )
        else:
            # Extract wavetables for each channel
            left_start = left_crossings[0]
            left_end = left_crossings[-1]
            left_wave = data[left_start:left_end, 0].copy()

            if right_crossings is not None and len(right_crossings) >= 2:
                right_start = right_crossings[0]
                right_end = right_crossings[-1]
                right_wave = data[right_start:right_end, 1].copy()
            else:
                # If no right crossings, use left crossings for right channel too
                right_wave = (
                    data[left_start:left_end, 1].copy()
                    if data.shape[1] > 1
                    else left_wave.copy()
                )

        # Find optimal size and resample both channels to match
        left_size = len(left_wave)
        right_size = len(right_wave)
        optimal_size = self._find_optimal_wavetable_size(left_size, right_size)

        # Resample both channels to optimal size
        left_resampled = self._hermite_resample(left_wave, optimal_size)
        right_resampled = self._hermite_resample(right_wave, optimal_size)

        # Combine into stereo wavetable
        stereo_wavetable = np.column_stack((left_resampled, right_resampled))

        if self.normalize:
            # Normalize each channel independently to prevent amplitude variations
            for ch in range(2):
                max_val = np.max(np.abs(stereo_wavetable[:, ch])) * 0.9
                if max_val > 0:
                    stereo_wavetable[:, ch] = stereo_wavetable[:, ch] / max_val

        return stereo_wavetable

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

    def _hermite_resample(self, data, target_length):
        """Resample data to target length using Hermite interpolation."""
        source_length = len(data)
        if source_length == target_length:
            return data.copy()

        resampled = np.zeros(target_length)
        scale = (source_length - 1) / (target_length - 1)

        for i in range(target_length):
            # Calculate source position
            pos = i * scale
            idx = int(pos)
            frac = pos - idx

            # Get neighboring samples for Hermite interpolation
            y0 = data[idx - 1] if idx > 0 else data[0]
            y1 = data[idx]
            y2 = data[idx + 1] if idx < source_length - 1 else data[-1]
            y3 = data[idx + 2] if idx < source_length - 2 else data[-1]

            resampled[i] = self._hermite_interpolate(y0, y1, y2, y3, frac)

        return resampled

    def _find_optimal_wavetable_size(self, left_size, right_size):
        """Find the optimal wavetable size that minimizes resampling error."""
        if right_size is None:  # Mono
            return left_size

        # Calculate resampling error for different target sizes
        min_size = min(left_size, right_size)
        max_size = max(left_size, right_size)

        # Test a range of sizes and find the one with minimum total resampling
        best_size = left_size
        min_error = float("inf")

        for size in range(min_size, max_size + 1):
            # Estimate resampling error as the sum of size differences
            error = abs(size - left_size) + abs(size - right_size)
            if error < min_error:
                min_error = error
                best_size = size

        return best_size

    def _get_wavetable_sample(self, wavetable, phase):
        """Get interpolated stereo sample from wavetable using Hermite interpolation."""
        wavetable_size = len(wavetable)
        idx = int(phase)
        frac = phase - idx

        # Handle stereo wavetables
        num_channels = wavetable.shape[1] if wavetable.ndim > 1 else 1
        result = np.zeros(num_channels)

        if idx < wavetable_size - 1:
            # Process each channel
            for ch in range(num_channels):
                if wavetable.ndim > 1:
                    # Get four points for Hermite interpolation
                    y0 = wavetable[idx - 1, ch] if idx > 0 else wavetable[-1, ch]
                    y1 = wavetable[idx, ch]
                    y2 = wavetable[idx + 1, ch]
                    y3 = (
                        wavetable[idx + 2, ch]
                        if idx < wavetable_size - 2
                        else wavetable[0, ch]
                    )
                else:
                    # Mono wavetable
                    y0 = wavetable[idx - 1] if idx > 0 else wavetable[-1]
                    y1 = wavetable[idx]
                    y2 = wavetable[idx + 1]
                    y3 = (
                        wavetable[idx + 2] if idx < wavetable_size - 2 else wavetable[0]
                    )

                result[ch] = self._hermite_interpolate(y0, y1, y2, y3, frac)
        else:
            if wavetable.ndim > 1:
                result = wavetable[idx]
            else:
                result[0] = wavetable[idx]

        return result

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

            # Calculate current sample (now returns stereo)
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

                # Crossfade between wavetables (now handles stereo)
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

            # Apply stereo sample to output
            if len(sample) == 2:
                output[i, 0] = sample[0]
                if self.channels > 1:
                    output[i, 1] = sample[1]
            else:
                # Mono sample - duplicate to all channels
                for ch in range(self.channels):
                    output[i, ch] = sample[0]

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
