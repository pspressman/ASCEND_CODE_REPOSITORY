from pathlib import Path
import subprocess
import sys

# Input directories with their corresponding output directories
PROCESS_CONFIGS = [
    {
        "input": Path("/Volumes/Databackup2025/ForLOCUS/CleanedSplicedAudio/CSA-Research"),
        "output": Path("/path/to/user/Desktop/TIMESTAMPED_CLEAN_OUTPUT/CSA-Research"),
        "skip_dirs": [
            "Mac_Audacity_10MinConvo_withCoordinatorSpeech",
            "Mac_Audacity_10MinuteConvo_withoutCoordinatorSpeech",
            "Mac_Audacity_Grandfather_Passage",
            "Mac_Audacity_Motor_Speech_Evaluation",
            "Mac_Audacity_Picnic_Description",
            "Mac_Audacity_Spontaneous_Speech"
        ]
    },
    {
        "input": Path("/Volumes/Databackup2025/ForLOCUS/CleanedSplicedAudio/Clinic_Plus_segmented"),
        "output": Path("/path/to/user/Desktop/TIMESTAMPED_CLEAN_OUTPUT/Clinic_Plus_segmented"),
        "skip_dirs": []
    }
]

# WhisperX parameters
MODEL = "large-v2"  # Can change to: tiny, base, small, medium, large-v1, large-v2, large-v3
DEVICE = "mps"  # Use "cpu" if no GPU, "mps" for Mac M1/M2/M3, "cuda" for NVIDIA GPU
BATCH_SIZE = 8  # Adjust based on GPU memory
COMPUTE_TYPE = "float16"  # Use "int8" for lower memory, "float16" for GPU

def process_directory(input_dir, output_base, skip_dirs=None):
    """Process all WAV files in a directory with WhisperX"""
    
    if skip_dirs is None:
        skip_dirs = []
    
    if not input_dir.exists():
        print(f"⚠ Directory does not exist: {input_dir}")
        return
    
    # Find all WAV files
    wav_files = list(input_dir.rglob("*.wav"))
    
    if not wav_files:
        print(f"⚠ No WAV files found in: {input_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"Processing directory: {input_dir}")
    print(f"Found {len(wav_files)} WAV files")
    if skip_dirs:
        print(f"Skipping subdirectories: {', '.join(skip_dirs)}")
    print(f"{'='*80}\n")
    
    # Create output base directory
    output_base.mkdir(parents=True, exist_ok=True)
    
    processed = 0
    errors = 0
    skipped = 0
    skipped_dir = 0
    
    for wav_file in wav_files:
        # Check if file is in a skip directory
        skip_this = False
        for skip_dir in skip_dirs:
            if skip_dir in wav_file.parts:
                print(f"⏭ Skipping (in excluded directory): {wav_file.relative_to(input_dir)}")
                skipped_dir += 1
                skip_this = True
                break
        
        if skip_this:
            continue
        
        # Get relative path to maintain directory structure
        rel_path = wav_file.relative_to(input_dir)
        output_dir = output_base / rel_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already processed
        json_output = output_dir / f"{wav_file.stem}.json"
        if json_output.exists():
            print(f"⏭ Skipping (already exists): {rel_path}")
            skipped += 1
            continue
        
        print(f"🎙 Processing: {rel_path}")
        
        # Build WhisperX command
        cmd = [
            "whisperx",
            str(wav_file),
            "--model", MODEL,
            "--device", DEVICE,
            "--batch_size", str(BATCH_SIZE),
            "--compute_type", COMPUTE_TYPE,
            "--output_dir", str(output_dir),
            "--output_format", "all",  # Generates json, srt, vtt, txt, tsv
            "--language", "en",  # Change if needed
            "--align_model", "WAV2VEC2_ASR_LARGE_LV60K_960H",  # For alignment/timestamps
            "--print_progress", "True"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            print(f"✓ Completed: {wav_file.stem}")
            processed += 1
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Error processing {wav_file.stem}:")
            print(f"  {e.stderr}")
            errors += 1
        except Exception as e:
            print(f"✗ Unexpected error processing {wav_file.stem}: {e}")
            errors += 1
    
    print(f"\n{'='*80}")
    print(f"Directory complete: {input_dir.name}")
    print(f"{'='*80}")
    print(f"Processed: {processed}")
    print(f"Skipped (already done): {skipped}")
    print(f"Skipped (excluded directories): {skipped_dir}")
    print(f"Errors: {errors}")
    print(f"Total WAV files found: {len(wav_files)}")
    print()

def main():
    """Process all configured directories"""
    
    print("WhisperX Batch Processor")
    print("="*80)
    print(f"Model: {MODEL}")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Compute type: {COMPUTE_TYPE}")
    print("="*80)
    
    for config in PROCESS_CONFIGS:
        try:
            print(f"\n📂 Input:  {config['input']}")
            print(f"📤 Output: {config['output']}")
            process_directory(
                config['input'], 
                config['output'],
                config.get('skip_dirs', [])
            )
        except KeyboardInterrupt:
            print("\n\n⚠ Processing interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n✗ Error processing directory {config['input']}: {e}\n")
    
    print("\n" + "="*80)
    print("ALL PROCESSING COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
