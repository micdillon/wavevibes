"""Audio file I/O utilities."""

import wave
from typing import Tuple

import numpy as np


def read_wave(filename: str) -> Tuple[np.ndarray, int, int]:
    """Read a WAV file and return audio data, sample rate, and bit depth.

    Args:
        filename: Path to WAV file

    Returns:
        audio_data: Audio samples as numpy array (n_samples, n_channels)
        sample_rate: Sample rate in Hz
        bit_depth: Bit depth (8, 16, 24, or 32)
    """
    with wave.open(filename, "rb") as wav:
        n_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        n_frames = wav.getnframes()

        # Read raw data
        raw_data = wav.readframes(n_frames)

        # Convert to numpy array
        if sample_width == 1:
            dtype = np.uint8
            max_val = 255
            audio_data = np.frombuffer(raw_data, dtype=dtype)
        elif sample_width == 2:
            dtype = np.int16
            max_val = 32768
            audio_data = np.frombuffer(raw_data, dtype=dtype)
        elif sample_width == 3:
            # 24-bit audio requires special handling
            # Each sample is 3 bytes, we need to convert to 32-bit integers
            max_val = 8388608  # 2^23
            total_samples = n_frames * n_channels
            audio_data = np.zeros(total_samples, dtype=np.int32)
            
            # Process each 3-byte sample
            for i in range(total_samples):
                byte_offset = i * 3
                # Read 3 bytes and convert to signed 32-bit integer
                # Little-endian: least significant byte first
                sample = (raw_data[byte_offset] | 
                         (raw_data[byte_offset + 1] << 8) | 
                         (raw_data[byte_offset + 2] << 16))
                
                # Sign extend from 24-bit to 32-bit
                if sample & 0x800000:  # If sign bit is set
                    sample |= 0xFF000000  # Extend sign to 32 bits
                    # Convert from unsigned to signed representation
                    sample = sample - 0x100000000
                
                audio_data[i] = sample
                
        elif sample_width == 4:
            dtype = np.int32
            max_val = 2147483648
            audio_data = np.frombuffer(raw_data, dtype=dtype)
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        # Reshape for multi-channel
        if n_channels > 1:
            audio_data = audio_data.reshape(-1, n_channels)

        # Convert to float [-1, 1]
        audio_data = audio_data.astype(np.float32)
        if sample_width == 1:
            audio_data = (audio_data - 128) / 128.0
        else:
            audio_data = audio_data / max_val

    bit_depth = sample_width * 8
    return audio_data, sample_rate, bit_depth


def write_wave(
    filename: str, audio_data: np.ndarray, sample_rate: int, sample_width: int = 2
):
    """Write audio data to a WAV file.

    Args:
        filename: Output WAV file path
        audio_data: Audio samples as numpy array (n_samples, n_channels) or (n_samples,)
        sample_rate: Sample rate in Hz
        sample_width: Bytes per sample (1, 2, 3, or 4)
    """
    # Ensure 2D array
    if audio_data.ndim == 1:
        audio_data = audio_data.reshape(-1, 1)

    n_channels = audio_data.shape[1]

    # Convert from float to integer
    if sample_width == 1:
        dtype = np.uint8
        max_val = 127
        audio_data = (audio_data * max_val + 128).clip(0, 255)
    elif sample_width == 2:
        dtype = np.int16
        max_val = 32767
        audio_data = (audio_data * max_val).clip(-32768, 32767)
    elif sample_width == 3:
        # 24-bit audio requires special handling
        max_val = 8388607  # 2^23 - 1
        audio_data = (audio_data * max_val).clip(-8388608, 8388607)
        audio_data = audio_data.astype(np.int32)
        
        # Convert 32-bit integers to 24-bit byte representation
        total_samples = audio_data.size
        byte_data = bytearray(total_samples * 3)
        
        # Flatten audio data for processing
        flat_audio = audio_data.flatten()
        
        for i in range(total_samples):
            sample = int(flat_audio[i])
            # Extract 3 bytes (little-endian)
            byte_offset = i * 3
            byte_data[byte_offset] = sample & 0xFF
            byte_data[byte_offset + 1] = (sample >> 8) & 0xFF
            byte_data[byte_offset + 2] = (sample >> 16) & 0xFF
            
        # Write WAV file with 24-bit data
        with wave.open(filename, "wb") as wav:
            wav.setnchannels(n_channels)
            wav.setsampwidth(sample_width)
            wav.setframerate(sample_rate)
            wav.writeframes(bytes(byte_data))
        return
        
    elif sample_width == 4:
        dtype = np.int32
        max_val = 2147483647
        audio_data = (audio_data * max_val).clip(-2147483648, 2147483647)
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    audio_data = audio_data.astype(dtype)

    # Write WAV file (for non-24-bit formats)
    with wave.open(filename, "wb") as wav:
        wav.setnchannels(n_channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_data.tobytes())
