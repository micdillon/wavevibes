#!/bin/bash
# Test script for real-time freezer

echo "Testing real-time freezer..."
echo "Commands to try:"
echo "  0.1 0.3 2.0    - Move to new position over 2 seconds"
echo "  status         - Show current status"
echo "  help           - Show all commands"
echo "  quit           - Exit"
echo ""

python scripts/freezer_rt.py test_files/synth1.wav --start-loc 0.2 --end-loc 0.3