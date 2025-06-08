#!/usr/bin/env python3
"""Real-time wavetable freezer with REPL interface."""

import argparse
import queue
import sys
import threading
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd

from wavevibes.io import read_wave
from freezer import Freezer


class FreezerRT(Freezer):
    """Real-time version of Freezer with dynamic parameter updates."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_loc = self.current_loc
        self.target_end_loc = self.end_loc
        self.transition_samples = 0
        self.transition_counter = 0
        self.is_transitioning = False
        
    def update_target(self, start: float, end: float, transition_time: float):
        """Update target locations with transition time."""
        self.target_loc = start
        self.target_end_loc = end
        self.transition_samples = int(transition_time * self.sample_rate)
        self.transition_counter = 0
        self.is_transitioning = True
        
    def process(self, block):
        """Process with dynamic parameter updates."""
        # Handle smooth transitions
        if self.is_transitioning:
            # Calculate transition progress
            progress = min(1.0, self.transition_counter / max(1, self.transition_samples))
            
            # Smoothly interpolate to new locations
            self.start_loc = self.start_loc + (self.target_loc - self.start_loc) * progress
            self.end_loc = self.end_loc + (self.target_end_loc - self.end_loc) * progress
            
            # Update counter
            self.transition_counter += block.shape[0]
            
            # Check if transition complete
            if self.transition_counter >= self.transition_samples:
                self.start_loc = self.target_loc
                self.end_loc = self.target_end_loc
                self.is_transitioning = False
                
        # Call parent process method
        return super().process(block)


class AudioStream:
    """Manages real-time audio output stream."""
    
    def __init__(self, freezer: FreezerRT, block_size: int = 512, latency: str = 'low'):
        self.freezer = freezer
        self.block_size = block_size
        self.latency = latency
        self.stream = None
        self.is_running = False
        
    def audio_callback(self, outdata, frames, time, status):
        """Audio callback for sounddevice."""
        if status:
            print(f"Audio callback status: {status}")
            
        # Generate audio
        block = np.zeros((frames, self.freezer.channels))
        processed = self.freezer.process(block)
        
        # Copy to output buffer
        outdata[:] = processed
        
    def start(self):
        """Start the audio stream."""
        self.stream = sd.OutputStream(
            samplerate=self.freezer.sample_rate,
            blocksize=self.block_size,
            channels=self.freezer.channels,
            callback=self.audio_callback,
            latency=self.latency
        )
        self.stream.start()
        self.is_running = True
        
    def stop(self):
        """Stop the audio stream."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.is_running = False


class REPLInterface:
    """REPL interface for controlling the freezer."""
    
    def __init__(self, freezer: FreezerRT):
        self.freezer = freezer
        self.running = True
        self.command_queue = queue.Queue()
        
    def parse_command(self, cmd: str) -> Optional[Tuple]:
        """Parse user command."""
        parts = cmd.strip().split()
        
        if not parts:
            return None
            
        command = parts[0].lower()
        
        if command in ['quit', 'exit', 'q']:
            return ('quit',)
        elif command == 'help':
            return ('help',)
        elif command == 'status':
            return ('status',)
        elif len(parts) == 2:
            # Two parameters: "location duration" format (move from current to location)
            try:
                location = float(parts[0])
                duration = float(parts[1])
                # Use current location as start
                return ('move', location, duration)
            except ValueError:
                return ('error', 'Invalid format. Use: <location> <duration>')
        elif len(parts) == 3:
            # Three parameters: "start end duration" format
            try:
                start = float(parts[0])
                end = float(parts[1])
                duration = float(parts[2])
                return ('set', start, end, duration)
            except ValueError:
                return ('error', 'Invalid format. Use: <start> <end> <duration>')
        else:
            return ('error', 'Unknown command. Type "help" for available commands.')
            
    def print_help(self):
        """Print help message."""
        print("\n=== Freezer RT Commands ===")
        print("<location> <duration>     - Move from current to location")
        print("<start> <end> <duration>  - Set new locations with transition")
        print("status                    - Show current parameters")
        print("help                      - Show this help")
        print("quit                      - Exit application")
        print("\nExamples:")
        print("  0.5 2.0       (move from current position to 0.5 over 2 seconds)")
        print("  0.2 0.8 3.0   (move from 0.2 to 0.8 over 3 seconds)\n")
        
    def print_status(self):
        """Print current status."""
        print(f"\n=== Current Status ===")
        print(f"Current location: {self.freezer.current_loc:.3f}")
        print(f"Start location: {self.freezer.start_loc:.3f}")
        print(f"End location: {self.freezer.end_loc:.3f}")
        print(f"Transitioning: {self.freezer.is_transitioning}")
        print(f"Wavetables cached: {len(self.freezer.wavetable_cache)}")
        print()
        
    def run(self):
        """Run the REPL."""
        print("\n=== Wavetable Freezer RT ===")
        print("Type 'help' for available commands")
        print()
        
        while self.running:
            try:
                # Get user input
                cmd = input("> ")
                
                # Parse command
                result = self.parse_command(cmd)
                
                if not result:
                    continue
                    
                action = result[0]
                
                if action == 'quit':
                    self.running = False
                    print("Exiting...")
                elif action == 'help':
                    self.print_help()
                elif action == 'status':
                    self.print_status()
                elif action == 'move':
                    _, location, duration = result
                    # Validate parameters
                    if 0 <= location <= 1 and duration > 0:
                        # Use current location as start
                        current = self.freezer.current_loc
                        self.freezer.update_target(current, location, duration)
                        print(f"Moving from {current:.3f} to {location:.3f} over {duration:.1f}s")
                    else:
                        print("Error: Location must be between 0 and 1, duration must be positive")
                elif action == 'set':
                    _, start, end, duration = result
                    # Validate parameters
                    if 0 <= start <= 1 and 0 <= end <= 1 and duration > 0:
                        self.freezer.update_target(start, end, duration)
                        print(f"Transitioning to [{start:.3f}, {end:.3f}] over {duration:.1f}s")
                    else:
                        print("Error: Locations must be between 0 and 1, duration must be positive")
                elif action == 'error':
                    print(f"Error: {result[1]}")
                    
            except KeyboardInterrupt:
                print("\nUse 'quit' to exit")
            except Exception as e:
                print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Real-time wavetable freezer')
    parser.add_argument('input_file', help='Input WAV file')
    parser.add_argument('--start-loc', type=float, default=0.2, help='Initial start location (0-1)')
    parser.add_argument('--end-loc', type=float, default=0.3, help='Initial end location (0-1)')
    parser.add_argument('--block-size', type=int, default=512, help='Audio block size')
    parser.add_argument('--latency', choices=['low', 'high'], default='low', help='Audio latency setting')
    parser.add_argument('--channels', type=int, default=2, help='Number of output channels')
    
    args = parser.parse_args()
    
    # Load audio file
    print(f"Loading {args.input_file}...")
    audio_data, samplerate, bit_depth = read_wave(args.input_file)
    
    # Create freezer
    freezer = FreezerRT(
        samplerate,
        args.block_size,
        args.channels,
        audio_data,
        start_loc=args.start_loc,
        end_loc=args.end_loc,
        n_crossings=16,
        interp_time=0.5,
        min_distance=0.01,
    )
    
    # Create and start audio stream
    audio_stream = AudioStream(freezer, args.block_size, args.latency)
    audio_stream.start()
    
    # Run REPL interface
    repl = REPLInterface(freezer)
    try:
        repl.run()
    finally:
        # Clean up
        audio_stream.stop()
        

if __name__ == "__main__":
    main()