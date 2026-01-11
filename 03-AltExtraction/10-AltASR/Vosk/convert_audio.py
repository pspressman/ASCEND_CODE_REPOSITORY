# convert_audio.py - save to Desktop
from pathlib import Path
import soundfile as sf
import numpy as np
from tqdm import tqdm

SOURCE_DIR = "/path/to/user/Desktop/AudioThreeTest/otherAudio"  # YOUR SOURCE
OUTPUT_DIR = "/path/to/volumes/Databackup2025/VoskWavs"  # LOCATION WITH SPACE

source_dir = Path(SOURCE_DIR)
output_dir = Path(OUTPUT_DIR)

audio_files = list(source_dir.rglob("*.wav"))
print(f"Converting {len(audio_files)} files to mono 16kHz 16-bit...")

for audio_file in tqdm(audio_files):
    try:
        # Load audio
        audio, sr = sf.read(str(audio_file))
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Preserve directory structure
        rel_path = audio_file.relative_to(source_dir)
        output_file = output_dir / rel_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as mono 16kHz 16-bit
        sf.write(str(output_file), audio, 16000, subtype='PCM_16')
        
    except Exception as e:
        print(f"\n✗ {audio_file.name}: {e}")

print("\n✓ Conversion complete!")