import argparse

from wavevibes import AudioProcessor, SimpleDelay


def main():
    parser = argparse.ArgumentParser(description="WaveVibes - Audio processing prototype")
    parser.add_argument("input", help="Input WAV file")
    parser.add_argument("output", help="Output WAV file")
    parser.add_argument("--delay-ms", type=float, default=250.0, help="Delay time in ms (default: 250)")
    parser.add_argument("--feedback", type=float, default=0.5, help="Feedback amount 0-0.95 (default: 0.5)")
    parser.add_argument("--mix", type=float, default=0.5, help="Dry/wet mix 0-1 (default: 0.5)")
    parser.add_argument("--block-size", type=int, default=512, help="Block size (default: 512)")
    
    args = parser.parse_args()
    
    # Create processor with algorithm factory
    # The algorithm will be created with the correct sample rate and channels from the input file
    processor = AudioProcessor(
        algorithm_factory=SimpleDelay,
        block_size=args.block_size,
        delay_ms=args.delay_ms,
        feedback=args.feedback,
        mix=args.mix
    )
    
    # Process file
    print(f"Processing {args.input} -> {args.output}")
    print(f"Delay: {args.delay_ms}ms, Feedback: {args.feedback}, Mix: {args.mix}")
    print(f"Block size: {args.block_size}")
    
    def progress(p):
        print(f"\rProgress: {p*100:.1f}%", end="", flush=True)
    
    processor.process_file(args.input, args.output, progress_callback=progress)
    print("\nDone!")


if __name__ == "__main__":
    main()
