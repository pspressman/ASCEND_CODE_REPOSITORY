from pathlib import Path
import shutil

# Source: where RTTMs currently are
SOURCE_DIR = Path("/path/to/volumes/video_research/ASCEND_PROCESSING/Audio/Audio/primary_path_output/CSA_RESEARCH_transcripts&OutputOriginal")

# Destination: where cleaned WAVs are now
DEST_DIR = Path("/path/to/user/Desktop/CleanedAudio_PC_NoiseReduced/CSA-Research")

def copy_rttm_files():
    """Copy RTTM files from source to destination, matching WAV file locations and preserving directory structure"""
    
    # Find all RTTM files in source
    rttm_files = list(SOURCE_DIR.rglob("*.rttm"))
    print(f"Found {len(rttm_files)} RTTM files in source")
    
    # Find all WAV files in destination
    wav_files = list(DEST_DIR.rglob("*.wav"))
    print(f"Found {len(wav_files)} WAV files in destination\n")
    
    # Create lookup by basename (removing "_cleaned" suffix if present)
    # Key: original basename, Value: wav file path
    wav_lookup = {}
    for wav in wav_files:
        stem = wav.stem
        # Remove "_cleaned" suffix if present to match with RTTM basename
        if stem.endswith("_cleaned"):
            original_stem = stem[:-8]  # Remove "_cleaned"
        else:
            original_stem = stem
        wav_lookup[original_stem] = wav
    
    copied = 0
    not_found = 0
    already_exist = 0
    
    for rttm_file in rttm_files:
        basename = rttm_file.stem
        
        if basename in wav_lookup:
            wav_file = wav_lookup[basename]
            # Place RTTM in same directory as corresponding WAV
            dest_rttm = wav_file.parent / f"{basename}.rttm"
            
            if dest_rttm.exists():
                print(f"⚠ Already exists: {basename}.rttm")
                already_exist += 1
            else:
                # Ensure destination directory exists
                dest_rttm.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(rttm_file, dest_rttm)
                print(f"✓ Copied: {basename}.rttm → {dest_rttm.relative_to(DEST_DIR)}")
                copied += 1
        else:
            print(f"✗ No matching WAV for: {basename}.rttm")
            not_found += 1
    
    print("\n" + "="*80)
    print("COPY COMPLETE")
    print("="*80)
    print(f"Copied: {copied}")
    print(f"Already existed: {already_exist}")
    print(f"No matching WAV: {not_found}")
    print(f"Total RTTM files processed: {len(rttm_files)}")

if __name__ == "__main__":
    copy_rttm_files()
