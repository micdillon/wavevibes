# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Installation
```bash
pip install -e .
```

### Running the Application
```bash
# Process audio with built-in delay effect
python main.py input.wav output.wav --delay-ms 300 --feedback 0.4 --mix 0.6

# Generate test audio files
python generate_test_files.py test.wav --type tone --frequency 440 --duration 2
python generate_test_files.py chirp.wav --type chirp --start-freq 100 --end-freq 2000
python generate_test_files.py noise.wav --type noise --amplitude 0.3

# Generate variety of example files
./generate_examples.sh
```

### Custom Algorithm Example
```bash
python example_custom_algorithm.py input.wav output.wav
```

## Architecture

WaveVibes is a Python module for prototyping real-time audio algorithms using block-based processing. The architecture simulates real-time processing by dividing audio into fixed-size blocks.

### Core Components

1. **Algorithm Base Class** (wavevibes/algorithms.py:7)
   - Abstract base class that all audio processing algorithms inherit from
   - Defines the interface: `process(block)` method and optional `reset()` method
   - Algorithms maintain state between blocks to enable effects like delays and filters

2. **AudioProcessor** (wavevibes/core.py:9)
   - Main processing engine that handles file I/O and block processing
   - Supports two initialization modes:
     - Algorithm factory: Creates algorithm with correct parameters from input file
     - Pre-initialized algorithm: Uses existing algorithm instance
   - Handles overlap-add processing for smoother results
   - Preserves input file bit-depth in output

3. **Audio I/O** (wavevibes/io.py)
   - Handles WAV file reading/writing with automatic format detection
   - Preserves bit-depth (16/24/32-bit) from input to output
   - Supports mono and stereo audio

### Processing Flow

1. Input WAV file is read, extracting sample rate, channels, and bit-depth
2. AudioProcessor creates/configures the algorithm based on input parameters
3. Audio is processed in fixed-size blocks (default 512 samples)
4. Optional overlap-add processing with Hanning window for smooth transitions
5. Output is written with same bit-depth as input

### Built-in Algorithms

- **PassThrough**: No processing (testing)
- **Gain**: Apply gain in dB
- **SimpleDelay**: Delay with feedback and mix control
- **SimpleLowpass**: One-pole lowpass filter (example in example_custom_algorithm.py)

### Creating Custom Algorithms

Custom algorithms inherit from the Algorithm base class and implement:
- `__init__`: Initialize with sample_rate, block_size, channels, plus custom parameters
- `process(block)`: Process a numpy array of shape (block_size, channels)
- `reset()`: Optional method to reset internal state

The block-based architecture allows algorithms to maintain state between blocks, enabling time-based effects like delays, reverbs, and filters while simulating real-time processing constraints.