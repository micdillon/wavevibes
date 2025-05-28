"""WaveVibes - A prototyping environment for real-time audio algorithms."""

from .core import AudioProcessor
from .io import read_wave, write_wave
from .algorithms import Algorithm, SimpleDelay, Gain

__version__ = "0.1.0"
__all__ = ["AudioProcessor", "read_wave", "write_wave", "Algorithm", "SimpleDelay", "Gain"]