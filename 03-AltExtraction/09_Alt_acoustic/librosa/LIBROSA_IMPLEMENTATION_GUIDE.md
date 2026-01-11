# Librosa Feature Extraction Scripts - Working Implementation

## Overview

These scripts extract 103 librosa-based acoustic features from audio files using pyannote RTTM diarization. They were developed for the ASCEND project to provide spectral and timbral analysis complementary to Parselmouth voice quality features.

**Key Achievement**: Successfully extracts features from audio files where `soundfile.read()` fails by using `librosa.load()` instead.

## Critical Implementation Detail

**IMPORTANT**: These scripts use `librosa.load()` instead of `soundfile.read()` for audio loading. This was essential because `soundfile.read()` failed on certain WAV files (returning only 2 samples instead of millions), while `librosa.load()` handled them correctly.

```python
# WORKS:
audio, sr = librosa.load(audio_path, sr=None, mono=True)

# FAILS on some files:
audio, sr = sf.read(audio_path)
```

## Scripts

### 1. librosa_per_turn.py
Extracts 103 features for each speaker turn identified in RTTM diarization.

**Output**: One CSV per audio file
- Filename: `{basename}_librosa_per_turn.csv`
- Rows: One per speaker turn
- Columns: `speaker`, `turn_start`, `turn_end`, `turn_duration` + 103 features

### 2. librosa_per_speaker_aggregated.py
Extracts 103 features aggregated across all turns for each speaker.

**Output**: Single CSV for all files
- Filename: Specified by `--output` argument
- Rows: One per speaker per file
- Columns: `filename`, `speaker`, `total_duration`, `num_turns` + 103 features

## Installation

### 1. Create Virtual Environment

```bash
# For Apple Silicon Macs (M1/M2/M3)
/opt/homebrew/bin/python3.10 -m venv /Users/peterpressman/MyDevelopment/Environments/librosa_extractor

# For Intel Macs
/usr/local/bin/python3.10 -m venv /Users/peterpressman/MyDevelopment/Environments/librosa_extractor
```

### 2. Activate Environment

```bash
source /Users/peterpressman/MyDevelopment/Environments/librosa_extractor/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install librosa==0.10.1
pip install soundfile==0.12.1
pip install pandas==2.0.3
pip install numpy==1.24.3
```

### 4. Verify Installation

```bash
python3 -c "import librosa; print(f'librosa: {librosa.__version__}')"
python3 -c "import pandas; print(f'pandas: {pandas.__version__}')"
python3 -c "import numpy; print(f'numpy: {numpy.__version__}')"
```

## Usage

### Input Structure

```
input_directory/
├── recording1.wav
├── recording1.rttm
├── recording2.wav
├── recording2.rttm
└── recording3.wav
    └── recording3.rttm
```

**Requirements**:
- WAV and RTTM files must have the same base name
- Must be in the same directory
- RTTM extension must be lowercase `.rttm`

### Per-Turn Extraction

```bash
python3 librosa_per_turn.py \
    --input /path/to/audio/directory \
    --output /path/to/output/directory
```

**Example**:
```bash
python3 librosa_per_turn.py \
    --input /Users/peterpressman/Desktop/AudioThreeTest/test \
    --output /Users/peterpressman/Desktop/AudioThreeTest/testResults
```

### Per-Speaker Aggregated Extraction

```bash
python3 librosa_per_speaker_aggregated.py \
    --input /path/to/audio/directory \
    --output /path/to/output/file.csv
```

**Example**:
```bash
python3 librosa_per_speaker_aggregated.py \
    --input /Users/peterpressman/Desktop/AudioThreeTest/test \
    --output /Users/peterpressman/Desktop/AudioThreeTest/testResults/librosa_aggregated.csv
```

## Features Extracted (103 Total)

### Spectral Features (48 features)

**Spectral Centroid** (4): mean, std, min, max
- Center of mass of the spectrum

**Spectral Bandwidth** (4): mean, std, min, max
- Width of the spectrum

**Spectral Rolloff** (4): mean, std, min, max
- Frequency below which 85% of spectral energy lies

**Spectral Contrast** (28): 7 bands × (mean, std, min, max)
- Peak-valley differences across 7 frequency bands

**Spectral Flatness** (4): mean, std, min, max
- Tonality vs. noise-like quality (0=tonal, 1=noise)

**Zero Crossing Rate** (4): mean, std, min, max
- Rate of signal sign changes (relates to noisiness)

### MFCCs (26 features)

**13 Coefficients** × (mean, std)
- Mel-Frequency Cepstral Coefficients
- Capture timbre and spectral envelope
- Standard features for speech recognition

### Chroma Features (24 features)

**12 Pitch Classes** × (mean, std)
- Energy in each pitch class (C, C#, D, ..., B)
- Independent of octave
- Useful for harmonic content analysis

### Energy Features (4 features)

**RMS Energy**: mean, std, min, max
- Root-mean-square energy
- Relates to loudness and intensity

### Custom Feature (1 feature)

**HF500**: Ratio of energy above/below 500 Hz
- High-frequency content indicator
- Useful for voice quality assessment

## Output Format

### Per-Turn CSV

```csv
speaker,turn_start,turn_end,turn_duration,spectral_centroid_mean,spectral_centroid_std,...
SPEAKER_00,0.229,1.485,1.256,1523.456,234.567,...
SPEAKER_00,2.250,3.303,1.053,1678.234,198.432,...
SPEAKER_01,102.946,108.651,5.705,1545.123,221.098,...
```

### Per-Speaker Aggregated CSV

```csv
filename,speaker,total_duration,num_turns,spectral_centroid_mean,spectral_centroid_std,...
4-010_21JUL2023_SpontSpeech,SPEAKER_00,0.798,2,1534.456,245.678,...
4-010_21JUL2023_SpontSpeech,SPEAKER_01,169.039,102,1512.789,238.901,...
4-010_21JUL2023_SpontSpeech,SPEAKER_02,48.365,17,1701.456,198.765,...
```

All numeric values are rounded to 3 decimal places.

## Processing Details

### Audio Loading
- Uses `librosa.load(audio_path, sr=None, mono=True)`
- Preserves original sample rate
- Automatically converts stereo to mono
- More robust than soundfile for various WAV formats

### Turn Filtering
- **Per-turn**: Minimum 0.1 seconds per turn
- **Per-speaker**: Minimum 0.5 seconds total across all turns
- Turns/speakers below threshold are skipped with warnings

### Feature Extraction
- Extracts from entire audio segment (not frame-by-frame)
- Uses librosa default parameters
- NaN values may appear if extraction fails

### Concatenation Method (Per-Speaker Aggregated)
1. All turns for each speaker are identified
2. Audio segments extracted chronologically
3. Segments concatenated in memory
4. Features extracted from concatenated audio

## Console Output

Example successful run:

```
SCRIPT STARTED
================================================================================
LIBROSA PER-SPEAKER AGGREGATED EXTRACTOR
================================================================================
Input:  /Users/peterpressman/Desktop/AudioThreeTest/test
Output: /Users/peterpressman/Desktop/AudioThreeTest/testResults/librosa_aggregated.csv
Found 1 WAV files
================================================================================

[1/1]
Processing: 4-010_21JUL2023_SpontSpeech
  Found 3 speakers
  Audio file: 15743700 samples, 357.00s @ 44100 Hz
  Speaker SPEAKER_01: 102 turns
    Turn 1: start=0.23s (10098 samples), end=1.49s (65488 samples), extracted=55390 samples
    Turn 2: start=2.25s (99225 samples), end=3.30s (145662 samples), extracted=46437 samples
    Turn 3: start=5.10s (224998 samples), end=6.88s (303628 samples), extracted=78630 samples
    Total duration: 169.04s (expected: 169.04s)
    Audio length: 7453115 samples, sr: 44100
  Speaker SPEAKER_02: 17 turns
    Total duration: 48.37s (expected: 48.37s)
    Audio length: 2132938 samples, sr: 44100
  Speaker SPEAKER_00: 2 turns
    Total duration: 0.80s (expected: 0.80s)
    Audio length: 35193 samples, sr: 44100
================================================================================
✅ Complete! Processed 1/1 files (0 skipped)
📊 Total speakers extracted: 3
📁 Output: /Users/peterpressman/Desktop/AudioThreeTest/testResults/librosa_aggregated.csv
================================================================================
```

## Troubleshooting

### "Audio file: 2 samples" or very small sample count

**Problem**: `soundfile.read()` failing to load audio properly
**Solution**: Scripts now use `librosa.load()` which handles this correctly
**Verification**: The corrected scripts will show millions of samples for typical recordings

### "No RTTM file" warnings

**Problem**: RTTM file not found
**Solutions**:
- Ensure RTTM has same base name as WAV
- Check extension is lowercase `.rttm`
- Verify both files are in same directory

### "No valid turns extracted" or "total audio too short"

**Problem**: All turns/speakers below minimum threshold
**Solutions**:
- Check if RTTM timestamps exceed audio duration
- Verify audio file is not truncated
- Consider lowering thresholds if analyzing very brief speech

### Memory issues

**Problem**: Large files causing memory errors
**Solutions**:
- Process files in smaller batches
- Increase system memory
- Consider splitting very long recordings

### IsADirectoryError when saving CSV

**Problem**: Output path points to existing directory
**Solution**: 
```bash
# Remove conflicting directory
rm -rf /path/to/conflicting/name.csv

# Or use different filename
python3 script.py --input ... --output /path/to/different_name.csv
```

## Comparison with Parselmouth

| Feature Type | Librosa (103) | Parselmouth (25) |
|-------------|---------------|------------------|
| **Spectral** | 48 (detailed) | 4 (moments) |
| **Voice Quality** | 0 | 4 (HNR, CPP, jitter, shimmer) |
| **MFCCs** | 26 | 0 |
| **Chroma** | 24 | 0 |
| **Energy** | 4 (RMS) | 2 (intensity) |
| **Formants** | 0 | 8 (F1, F2 + bandwidths) |
| **Prosody** | 0 | 5 (rates, pauses) |
| **Pitch** | 0 | 2 (F0 statistics) |

**Key Insight**: Librosa and Parselmouth provide complementary acoustic perspectives. Librosa focuses on spectral/timbral content, while Parselmouth focuses on voice quality and prosody.

## ASCEND Project Context

These scripts are part of the ASCEND (Automated Speech Comparison Engine for Neurocognitive Detection) framework, which systematically compares different feature extraction methods for speech-based dementia detection.

**Purpose**:
- Third acoustic toolkit (alongside eGeMAPS and Parselmouth)
- Tests spectral/timbral features for disease discrimination
- Validates MFCC-based approaches common in machine learning
- Provides complementary feature space to voice quality measures

**Integration**:
- Uses same RTTM diarization as Parselmouth scripts
- Compatible with multi-ASR pipeline (WhisperX, Faster-Whisper, Vosk)
- Supports de-identification validation workflow
- Enables cross-method performance comparison

## Version History

- **v1.0** (2025-10-21): Initial release
  - 103 librosa-based features
  - Per-turn and per-speaker aggregated extraction
  - Uses `librosa.load()` for robust audio loading
  - Consistent with Parselmouth script structure

## Future Use

To use the environment later:

```bash
# Activate
source /Users/peterpressman/MyDevelopment/Environments/librosa_extractor/bin/activate

# Run scripts...

# Deactivate when done
deactivate
```

## Citation

If using these features in research, please cite:

**Librosa:**
```
McFee, B., Raffel, C., Liang, D., Ellis, D. P., McVicar, M., Battenberg, E., & Nieto, O. (2015). 
librosa: Audio and music signal analysis in python. 
In Proceedings of the 14th python in science conference (Vol. 8, pp. 18-25).
```

**Pyannote (for diarization):**
```
Bredin, H., Yin, R., Coria, J. M., Gelly, G., Korshunov, P., Lavechin, M., ... & Gill, M. P. (2020). 
Pyannote. audio: neural building blocks for speaker diarization. 
In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) 
(pp. 7124-7128). IEEE.
```

## Support

For issues:
1. Verify virtual environment is activated
2. Check all dependencies are installed correctly
3. Ensure RTTM and WAV files have matching names
4. Review console output for specific error messages
5. Confirm audio files load correctly with `librosa.load()`

## Success Criteria

✅ Scripts load audio files correctly (millions of samples, not 2)
✅ Extracts features from all valid turns/speakers
✅ Produces CSV files with 103 features + metadata
✅ Console shows expected vs actual durations matching
✅ Works with same RTTM files as other ASCEND scripts
