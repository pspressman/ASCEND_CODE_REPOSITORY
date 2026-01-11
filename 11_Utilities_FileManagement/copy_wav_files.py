#!/usr/bin/env python3
"""
Copy all WAV files from source directory to destination,
preserving the original folder structure.
"""

import shutil
from pathlib import Path
from tqdm import tqdm

def copy_wav_files_with_structure(source_dir, dest_dir):
    """
    Recursively find and copy all WAV files from source to destination,
    maintaining the directory structure.
    
    Args:
        source_dir: Source directory path
        dest_dir: Destination directory path
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    # Verify source directory exists
    if not source_path.exists():
        print(f"Error: Source directory does not exist: {source_path}")
        return
    
    # Find all WAV files recursively
    wav_files = list(source_path.rglob("*.wav"))
    
    if not wav_files:
        print("No WAV files found in source directory.")
        return
    
    print(f"Found {len(wav_files)} WAV files to copy")
    
    copied_count = 0
    skipped_count = 0
    error_count = 0
    
    for wav_file in tqdm(wav_files, desc="Copying WAV files"):
        try:
            # Calculate relative path from source directory
            relative_path = wav_file.relative_to(source_path)
            
            # Create destination path maintaining structure
            dest_file_path = dest_path / relative_path
            
            # Check if file already exists
            if dest_file_path.exists():
                print(f"\nSkipping (already exists): {relative_path}")
                skipped_count += 1
                continue
            
            # Create parent directories if they don't exist
            dest_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy the file
            shutil.copy2(wav_file, dest_file_path)
            copied_count += 1
            
        except Exception as e:
            print(f"\nError copying {wav_file}: {e}")
            error_count += 1
            continue
    
    # Summary
    print("\n" + "="*50)
    print("COPY SUMMARY")
    print("="*50)
    print(f"Total WAV files found: {len(wav_files)}")
    print(f"Successfully copied: {copied_count}")
    print(f"Skipped (already exist): {skipped_count}")
    print(f"Errors: {error_count}")
    print("="*50)

if __name__ == "__main__":
    # Define source and destination directories
    source_directory = "/path/to/volumes/video_research/conversational speech 18-0456/Participant Data and Forms"
    destination_directory = "/path/to/volumes/video_research/CSAND_AUDIO/OriginalAudio"
    
    print(f"Source: {source_directory}")
    print(f"Destination: {destination_directory}")
    print()
    
    copy_wav_files_with_structure(source_directory, destination_directory)
