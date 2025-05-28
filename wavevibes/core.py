"""Core audio processing engine."""

import numpy as np
from typing import Optional, Callable, Dict, Any
from .algorithms import Algorithm
from .io import read_wave, write_wave


class AudioProcessor:
    """Main audio processor that handles block-based processing."""
    
    def __init__(self, algorithm_factory: Callable = None, algorithm: Algorithm = None, 
                 block_size: int = 512, **algorithm_params):
        """Initialize the audio processor.
        
        Args:
            algorithm_factory: Factory function to create algorithm with signature
                              (sample_rate, block_size, channels, **kwargs) -> Algorithm
            algorithm: Pre-initialized algorithm instance (use this OR algorithm_factory)
            block_size: Size of audio blocks to process
            **algorithm_params: Additional parameters to pass to algorithm factory
        """
        if algorithm_factory and algorithm:
            raise ValueError("Provide either algorithm_factory or algorithm, not both")
        if not algorithm_factory and not algorithm:
            raise ValueError("Must provide either algorithm_factory or algorithm")
            
        self.algorithm_factory = algorithm_factory
        self.algorithm = algorithm
        self.block_size = block_size
        self.algorithm_params = algorithm_params
    
    def process_file(self, input_file: str, output_file: str, 
                    overlap: float = 0.0, progress_callback: Optional[callable] = None):
        """Process an audio file block by block.
        
        Args:
            input_file: Path to input WAV file
            output_file: Path to output WAV file
            overlap: Overlap factor (0.0 to 0.9) for overlapping blocks
            progress_callback: Optional callback function(progress: float) for progress updates
        """
        # Read input file
        audio_data, sample_rate, bit_depth = read_wave(input_file)
        
        # Ensure 2D array
        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(-1, 1)
        
        n_samples, n_channels = audio_data.shape
        
        # Create algorithm if using factory
        if self.algorithm_factory:
            self.algorithm = self.algorithm_factory(
                sample_rate, self.block_size, n_channels, **self.algorithm_params
            )
        else:
            # Update existing algorithm parameters if they don't match
            if (self.algorithm.sample_rate != sample_rate or 
                self.algorithm.channels != n_channels or
                self.algorithm.block_size != self.block_size):
                # Warn user about parameter mismatch
                print(f"Warning: Algorithm initialized with different parameters than input file")
                print(f"  Algorithm: {self.algorithm.sample_rate}Hz, {self.algorithm.channels}ch")
                print(f"  File: {sample_rate}Hz, {n_channels}ch, {bit_depth}-bit")
        
        self.algorithm.reset()
        
        # Calculate hop size based on overlap
        hop_size = int(self.block_size * (1 - overlap))
        
        # Prepare output buffer
        output_data = np.zeros_like(audio_data)
        
        # Process blocks
        position = 0
        block_count = 0
        total_blocks = (n_samples - self.block_size) // hop_size + 1
        
        while position + self.block_size <= n_samples:
            # Extract block
            block = audio_data[position:position + self.block_size]
            
            # Process block
            processed_block = self.algorithm.process(block)
            
            # Handle overlapping output
            if overlap > 0:
                # Apply window for smooth overlap-add
                window = np.hanning(self.block_size).reshape(-1, 1)
                processed_block *= window
                
                # Add to output with overlap
                output_data[position:position + self.block_size] += processed_block
            else:
                # Non-overlapping: direct copy
                output_data[position:position + self.block_size] = processed_block
            
            # Update position
            position += hop_size
            block_count += 1
            
            # Progress callback
            if progress_callback:
                progress = block_count / total_blocks
                progress_callback(progress)
        
        # Process remaining samples if any
        if position < n_samples:
            remaining = n_samples - position
            block = np.zeros((self.block_size, n_channels))
            block[:remaining] = audio_data[position:]
            
            processed_block = self.algorithm.process(block)
            output_data[position:] = processed_block[:remaining]
        
        # Normalize output if using overlap-add
        if overlap > 0:
            # Compensate for windowing gain
            output_data /= (overlap + 1)
        
        # Write output file with same bit depth as input
        sample_width = bit_depth // 8
        write_wave(output_file, output_data, sample_rate, sample_width)
    
    def process_stream(self, audio_block: np.ndarray) -> np.ndarray:
        """Process a single block of audio (for real-time applications).
        
        Args:
            audio_block: Audio block to process
            
        Returns:
            Processed audio block
        """
        return self.algorithm.process(audio_block)