from pathlib import Path
import shutil

# Source: where RTTMs currently are
SOURCE_DIR = Path("/path/to/user/Desktop/AudioThreeTest/otherAudio")

# Destination: where WAVs are now
DEST_DIR = Path("/path/to/volumes/Databackup2025/VoskWavs")

def copy_rttm_files():
    """Copy RTTM files from source to destination, matching WAV file locations"""
    
    # Find all RTTM files in source
    rttm_files = list(SOURCE_DIR.rglob("*.rttm"))
    print(f"Found {len(rttm_files)} RTTM files in source")
    
    # Find all WAV files in destination
    wav_files = list(DEST_DIR.rglob("*.wav"))
    print(f"Found {len(wav_files)} WAV files in destination\n")
    
    # Create lookup by basename
    wav_lookup = {wav.stem: wav for wav in wav_files}
    
    copied = 0
    not_found = 0
    already_exist = 0
    
    for rttm_file in rttm_files:
        basename = rttm_file.stem
        
        if basename in wav_lookup:
            wav_file = wav_lookup[basename]
            dest_rttm = wav_file.parent / f"{basename}.rttm"
            
            if dest_rttm.exists():
                already_exist += 1
            else:
                shutil.copy2(rttm_file, dest_rttm)
                print(f"✓ Copied: {basename}.rttm")
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

if __name__ == "__main__":
    copy_rttm_files()