# WaveVibes

A simple Python module for prototyping real-time audio algorithms with block-based processing.

## Features

- Block-based audio processing (simulates real-time processing)
- Reads sample rate and bit-depth from input WAV files
- Preserves bit-depth in output files
- Simple algorithm interface for easy prototyping
- Support for mono and stereo audio
- Optional overlap-add processing
- Progress tracking
- Test file generator for creating various audio signals

## Installation

```bash
pip install -e .
```

## Usage

### Command Line

Process a file with the built-in delay effect:

```bash
python main.py input.wav output.wav --delay-ms 300 --feedback 0.4 --mix 0.6
```

### Generate Test Files

Generate various test signals for algorithm testing:

```bash
# Pure tone
python generate_test_files.py test.wav --type tone --frequency 440 --duration 2

# Linear chirp (frequency sweep)
python generate_test_files.py chirp.wav --type chirp --start-freq 100 --end-freq 2000

# Logarithmic chirp
python generate_test_files.py chirp_log.wav --type chirp --start-freq 100 --end-freq 5000 --chirp-method logarithmic

# White noise
python generate_test_files.py noise.wav --type noise --amplitude 0.3

# With envelope
python generate_test_files.py tone_envelope.wav --type tone --frequency 880 --envelope

# Stereo
python generate_test_files.py stereo.wav --type tone --frequency 440 --channels 2

# Different bit depths
python generate_test_files.py test_24bit.wav --type tone --bit-depth 24
```

Run `./generate_examples.sh` to create a variety of test files.

### Creating Custom Algorithms

1. Inherit from the `Algorithm` base class:

```python
from wavevibes import Algorithm
import numpy as np

class MyEffect(Algorithm):
    def __init__(self, sample_rate, block_size, channels, my_param=1.0):
        super().__init__(sample_rate, block_size, channels)
        self.my_param = my_param
    
    def process(self, block: np.ndarray) -> np.ndarray:
        # Process the audio block
        return block * self.my_param
```

2. Use with AudioProcessor:

```python
from wavevibes import AudioProcessor

# Method 1: Using algorithm factory (recommended)
# Algorithm is created with correct parameters from input file
processor = AudioProcessor(
    algorithm_factory=MyEffect,
    block_size=512,
    my_param=0.5
)

# Method 2: Pre-initialized algorithm
algorithm = MyEffect(44100, 512, 2, my_param=0.5)
processor = AudioProcessor(algorithm=algorithm, block_size=512)

# Process file
processor.process_file("input.wav", "output.wav")
```

## Built-in Algorithms

- `PassThrough`: No processing (for testing)
- `Gain`: Apply gain in dB
- `SimpleDelay`: Delay effect with feedback and mix control
- `SimpleLowpass`: One-pole lowpass filter (see example_custom_algorithm.py)

## Architecture

- Algorithms process audio in fixed-size blocks
- Input files are automatically analyzed for sample rate, channels, and bit-depth
- Output files preserve the input bit-depth
- Supports overlap-add processing for smoother results