#!/bin/bash

echo "Generating example test files..."

# Create test_files directory
mkdir -p test_files

# Generate pure tones
echo "1. Generating pure tones..."
python generate_test_files.py test_files/tone_440hz.wav --type tone --frequency 440 --duration 2
python generate_test_files.py test_files/tone_1khz_stereo.wav --type tone --frequency 1000 --duration 1 --channels 2
python generate_test_files.py test_files/tone_with_envelope.wav --type tone --frequency 880 --duration 2 --envelope --attack 0.1 --sustain 0.0 --release 0.2

# Generate chirps
echo "2. Generating chirps..."
python generate_test_files.py test_files/chirp_linear.wav --type chirp --start-freq 100 --end-freq 2000 --duration 3
python generate_test_files.py test_files/chirp_log.wav --type chirp --start-freq 100 --end-freq 5000 --duration 3 --chirp-method logarithmic
python generate_test_files.py test_files/chirp_with_envelope.wav --type chirp --start-freq 200 --end-freq 800 --duration 2 --envelope --attack 0.1 --sustain 0.0 --release 0.2

# Generate noise
echo "3. Generating noise..."
python generate_test_files.py test_files/white_noise.wav --type noise --duration 2 --amplitude 0.3

# Generate different bit depths
echo "4. Generating different bit depths..."
python generate_test_files.py test_files/tone_8bit.wav --type tone --frequency 440 --duration 1 --bit-depth 8
python generate_test_files.py test_files/tone_16bit.wav --type tone --frequency 440 --duration 1 --bit-depth 16
python generate_test_files.py test_files/tone_24bit.wav --type tone --frequency 440 --duration 1 --bit-depth 24

# Generate silence (useful for testing)
echo "5. Generating silence..."
python generate_test_files.py test_files/silence.wav --type silence --duration 1

echo "Done! Test files created in test_files/"