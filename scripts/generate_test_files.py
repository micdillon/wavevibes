#!/usr/bin/env python3
"""Generate test WAV files for testing audio algorithms."""

import argparse
import numpy as np
from wavevibes.io import write_wave


def generate_pure_tone(frequency, duration, sample_rate, amplitude=0.5):
    """Generate a pure sine wave tone.
    
    Args:
        frequency: Frequency in Hz
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        amplitude: Amplitude (0-1)
    
    Returns:
        Audio data as numpy array
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    return signal


def generate_chirp(start_freq, end_freq, duration, sample_rate, amplitude=0.5, method='linear'):
    """Generate a frequency sweep (chirp).
    
    Args:
        start_freq: Starting frequency in Hz
        end_freq: Ending frequency in Hz
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        amplitude: Amplitude (0-1)
        method: 'linear' or 'logarithmic' frequency sweep
    
    Returns:
        Audio data as numpy array
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    if method == 'linear':
        # Linear frequency sweep
        instantaneous_freq = start_freq + (end_freq - start_freq) * t / duration
        phase = 2 * np.pi * (start_freq * t + (end_freq - start_freq) * t**2 / (2 * duration))
    elif method == 'logarithmic':
        # Logarithmic frequency sweep
        if start_freq <= 0 or end_freq <= 0:
            raise ValueError("Frequencies must be positive for logarithmic sweep")
        log_start = np.log(start_freq)
        log_end = np.log(end_freq)
        instantaneous_freq = np.exp(log_start + (log_end - log_start) * t / duration)
        phase = 2 * np.pi * start_freq * duration / (log_end - log_start) * (np.exp((log_end - log_start) * t / duration) - 1)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    signal = amplitude * np.sin(phase)
    return signal


def generate_white_noise(duration, sample_rate, amplitude=0.5):
    """Generate white noise.
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        amplitude: Amplitude (0-1)
    
    Returns:
        Audio data as numpy array
    """
    n_samples = int(sample_rate * duration)
    signal = amplitude * (2 * np.random.random(n_samples) - 1)
    return signal


def generate_silence(duration, sample_rate):
    """Generate silence.
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
    
    Returns:
        Audio data as numpy array
    """
    n_samples = int(sample_rate * duration)
    return np.zeros(n_samples)


def apply_envelope(signal, sample_rate, attack=0.01, decay=0.01, sustain_level=1.0, release=0.05):
    """Apply an ADSR envelope to a signal.
    
    Args:
        signal: Input signal
        sample_rate: Sample rate in Hz
        attack: Attack time in seconds
        decay: Decay time in seconds
        sustain_level: Sustain level (0-1)
        release: Release time in seconds
    
    Returns:
        Signal with envelope applied
    """
    n_samples = len(signal)
    duration = n_samples / sample_rate
    
    # Calculate sample counts for each phase
    attack_samples = int(attack * sample_rate)
    decay_samples = int(decay * sample_rate)
    release_samples = int(release * sample_rate)
    sustain_samples = n_samples - attack_samples - decay_samples - release_samples
    
    if sustain_samples < 0:
        # Signal too short for full envelope
        total_env_time = attack + decay + release
        if duration < total_env_time:
            # Scale envelope times proportionally
            scale = duration / total_env_time
            attack_samples = int(attack * scale * sample_rate)
            decay_samples = int(decay * scale * sample_rate)
            release_samples = int(release * scale * sample_rate)
            sustain_samples = 0
    
    # Create envelope
    envelope = np.ones(n_samples)
    
    # Attack
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    
    # Decay
    if decay_samples > 0:
        start_idx = attack_samples
        end_idx = start_idx + decay_samples
        envelope[start_idx:end_idx] = np.linspace(1, sustain_level, decay_samples)
    
    # Sustain
    if sustain_samples > 0:
        start_idx = attack_samples + decay_samples
        end_idx = start_idx + sustain_samples
        envelope[start_idx:end_idx] = sustain_level
    
    # Release
    if release_samples > 0:
        start_idx = n_samples - release_samples
        envelope[start_idx:] = np.linspace(sustain_level, 0, release_samples)
    
    return signal * envelope


def main():
    parser = argparse.ArgumentParser(description="Generate test WAV files")
    parser.add_argument("output", help="Output WAV file path")
    parser.add_argument("--type", choices=['tone', 'chirp', 'noise', 'silence'], 
                       default='tone', help="Type of signal to generate")
    
    # Common parameters
    parser.add_argument("--duration", type=float, default=1.0, 
                       help="Duration in seconds (default: 1.0)")
    parser.add_argument("--sample-rate", type=int, default=44100, 
                       help="Sample rate in Hz (default: 44100)")
    parser.add_argument("--bit-depth", type=int, choices=[8, 16, 24, 32], default=16,
                       help="Bit depth (default: 16)")
    parser.add_argument("--channels", type=int, choices=[1, 2], default=1,
                       help="Number of channels (default: 1)")
    parser.add_argument("--amplitude", type=float, default=0.5,
                       help="Amplitude 0-1 (default: 0.5)")
    
    # Tone parameters
    parser.add_argument("--frequency", type=float, default=440.0,
                       help="Frequency in Hz for pure tone (default: 440)")
    
    # Chirp parameters
    parser.add_argument("--start-freq", type=float, default=100.0,
                       help="Start frequency for chirp (default: 100)")
    parser.add_argument("--end-freq", type=float, default=1000.0,
                       help="End frequency for chirp (default: 1000)")
    parser.add_argument("--chirp-method", choices=['linear', 'logarithmic'], default='linear',
                       help="Chirp method (default: linear)")
    
    # Envelope parameters
    parser.add_argument("--envelope", action='store_true',
                       help="Apply ADSR envelope to signal")
    parser.add_argument("--attack", type=float, default=0.01,
                       help="Envelope attack time in seconds (default: 0.01)")
    parser.add_argument("--decay", type=float, default=0.01,
                       help="Envelope decay time in seconds (default: 0.01)")
    parser.add_argument("--sustain", type=float, default=1.0,
                       help="Envelope sustain level 0-1 (default: 1.0)")
    parser.add_argument("--release", type=float, default=0.05,
                       help="Envelope release time in seconds (default: 0.05)")
    
    args = parser.parse_args()
    
    # Generate signal based on type
    if args.type == 'tone':
        print(f"Generating {args.duration}s pure tone at {args.frequency}Hz...")
        signal = generate_pure_tone(args.frequency, args.duration, 
                                   args.sample_rate, args.amplitude)
    
    elif args.type == 'chirp':
        print(f"Generating {args.duration}s {args.chirp_method} chirp from {args.start_freq}Hz to {args.end_freq}Hz...")
        signal = generate_chirp(args.start_freq, args.end_freq, args.duration,
                               args.sample_rate, args.amplitude, args.chirp_method)
    
    elif args.type == 'noise':
        print(f"Generating {args.duration}s white noise...")
        signal = generate_white_noise(args.duration, args.sample_rate, args.amplitude)
    
    elif args.type == 'silence':
        print(f"Generating {args.duration}s silence...")
        signal = generate_silence(args.duration, args.sample_rate)
    
    # Apply envelope if requested
    if args.envelope and args.type != 'silence':
        print("Applying ADSR envelope...")
        signal = apply_envelope(signal, args.sample_rate, 
                               args.attack, args.decay, args.sustain, args.release)
    
    # Handle stereo
    if args.channels == 2:
        if args.type == 'tone':
            # Create stereo with slightly different frequencies for interesting effect
            signal_r = generate_pure_tone(args.frequency * 1.01, args.duration,
                                         args.sample_rate, args.amplitude)
            if args.envelope:
                signal_r = apply_envelope(signal_r, args.sample_rate,
                                        args.attack, args.decay, args.sustain, args.release)
            signal = np.column_stack((signal, signal_r))
        else:
            # Duplicate mono to stereo
            signal = np.column_stack((signal, signal))
    
    # Write output file
    sample_width = args.bit_depth // 8
    write_wave(args.output, signal, args.sample_rate, sample_width)
    
    print(f"Generated {args.output}")
    print(f"  Type: {args.type}")
    print(f"  Duration: {args.duration}s")
    print(f"  Sample rate: {args.sample_rate}Hz")
    print(f"  Bit depth: {args.bit_depth}-bit")
    print(f"  Channels: {args.channels}")
    if args.type == 'tone':
        print(f"  Frequency: {args.frequency}Hz")
    elif args.type == 'chirp':
        print(f"  Frequency range: {args.start_freq}Hz - {args.end_freq}Hz ({args.chirp_method})")


if __name__ == "__main__":
    main()