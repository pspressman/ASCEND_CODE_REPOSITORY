# check_audio_format.py
import wave
from pathlib import Path
from collections import Counter

AUDIO_DIR = "/path/to/volumes/Databackup2025/VoskWavs"

audio_dir = Path(AUDIO_DIR)
audio_files = list(audio_dir.rglob("*.wav"))

print(f"Checking {len(audio_files)} files...\n")

formats = []
for audio_file in audio_files:
    try:
        wf = wave.open(str(audio_file), "rb")
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        wf.close()
        
        formats.append({
            'channels': channels,
            'sampwidth': sampwidth,
            'framerate': framerate
        })
    except Exception as e:
        print(f"Error reading {audio_file.name}: {e}")

# Summary
print("Format Summary:")
print(f"Channels: {Counter(f['channels'] for f in formats)}")
print(f"Sample Width (bytes): {Counter(f['sampwidth'] for f in formats)}")
print(f"Sample Rates: {Counter(f['framerate'] for f in formats)}")

# Check if conversion needed
needs_conversion = any(
    f['channels'] != 1 or 
    f['sampwidth'] != 2 or 
    f['framerate'] not in [8000, 16000, 32000, 48000]
    for f in formats
)

print(f"\nNeeds conversion: {'YES' if needs_conversion else 'NO'}")