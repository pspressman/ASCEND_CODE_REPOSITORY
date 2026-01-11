# Librosa Per-Turn Feature Extractor

## Overview

This script extracts 103 librosa-based acoustic features for each speaker turn identified by pyannote diarization. It processes audio files alongside their RTTM diarization files and produces one CSV per audio file containing per-turn features.

## Requirements

### Python Version
- Python 3.8 or higher

### Dependencies

```bash
pip install librosa==0.10.1
pip install soundfile==0.12.1
pip install pandas==2.0.3
pip install numpy==1.24.3
```

Or install all at once:
```bash
pip install librosa soundfile pandas numpy
```

## Features Extracted (103 Total)

### 1. Spectral Features (48 features)

**Spectral Centroid (4 features)**
- `spectral_centroid_mean` - Average center of mass of spectrum
- `spectral_centroid_std` - Variability of spectral center
- `spectral_centroid_min` - Minimum spectral center
- `spectral_centroid_max` - Maximum spectral center

**Spectral Bandwidth (4 features)**
- `spectral_bandwidth_mean` - Average width of spectrum
- `spectral_bandwidth_std` - Variability of spectral width
- `spectral_bandwidth_min` - Minimum spectral width
- `spectral_bandwidth_max` - Maximum spectral width

**Spectral Rolloff (4 features)**
- `spectral_rolloff_mean` - Average frequency below which 85% of energy lies
- `spectral_rolloff_std` - Variability of rolloff frequency
- `spectral_rolloff_min` - Minimum rolloff frequency
- `spectral_rolloff_max` - Maximum rolloff frequency

**Spectral Contrast (28 features - 7 bands × 4 statistics)**
- For each of 7 frequency bands:
  - `spectral_contrast_band{1-7}_mean` - Average peak-valley difference
  - `spectral_contrast_band{1-7}_std` - Variability of contrast
  - `spectral_contrast_band{1-7}_min` - Minimum contrast
  - `spectral_contrast_band{1-7}_max` - Maximum contrast

**Spectral Flatness (4 features)**
- `spectral_flatness_mean` - Average tonality vs. noise-like quality
- `spectral_flatness_std` - Variability of flatness
- `spectral_flatness_min` - Minimum flatness
- `spectral_flatness_max` - Maximum flatness

**Zero Crossing Rate (4 features)**
- `zero_crossing_rate_mean` - Average rate of signal sign changes
- `zero_crossing_rate_std` - Variability of crossing rate
- `zero_crossing_rate_min` - Minimum crossing rate
- `zero_crossing_rate_max` - Maximum crossing rate

### 2. MFCCs - Mel-Frequency Cepstral Coefficients (26 features)

**13 Coefficients × 2 Statistics**
- `mfcc{1-13}_mean` - Average of each MFCC coefficient
- `mfcc{1-13}_std` - Standard deviation of each coefficient

MFCCs capture the timbre and spectral envelope of speech.

### 3. Chroma Features (24 features)

**12 Pitch Classes × 2 Statistics**
- `chroma{1-12}_mean` - Average energy in each pitch class
- `chroma{1-12}_std` - Variability in each pitch class

Chroma features represent the 12 different pitch classes (C, C#, D, etc.) independent of octave.

### 4. Energy Features (4 features)

**RMS Energy**
- `rms_energy_mean` - Average root-mean-square energy
- `rms_energy_std` - Variability of energy
- `rms_energy_min` - Minimum energy
- `rms_energy_max` - Maximum energy

### 5. Custom Acoustic Feature (1 feature)

**HF500**
- `HF500` - Ratio of spectral energy above 500 Hz vs. below 500 Hz
  - Higher values indicate more high-frequency content
  - Useful for voice quality assessment

## Usage

### Basic Usage

```bash
python3 librosa_per_turn.py --input /path/to/wavs --output /path/to/output/dir
```

### Arguments

- `--input` (required): Directory containing WAV files and their corresponding RTTM files
- `--output` (required): Directory where output CSV files will be saved

### Example

```bash
python3 librosa_per_turn.py \
    --input /data/ascend/audio \
    --output /data/ascend/features/librosa_per_turn
```

## Input Requirements

### File Structure

Your input directory must contain:
- WAV files (any sample rate, mono or stereo)
- Corresponding RTTM files with same base name

```
input_directory/
├── participant001_task.wav
├── participant001_task.rttm
├── participant002_task.wav
├── participant002_task.rttm
└── ...
```

### RTTM Format

Standard pyannote diarization format:
```
SPEAKER filename 1 0.000 2.350 <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER filename 1 2.450 1.820 <NA> <NA> SPEAKER_01 <NA> <NA>
```

Fields: `SPEAKER filename channel start_time duration _ _ speaker_id _ _`

## Output Format

### File Naming

One CSV per audio file: `{basename}_librosa_per_turn.csv`

Example: `participant001_task.wav` → `participant001_task_librosa_per_turn.csv`

### CSV Structure

**Metadata Columns:**
- `speaker` - Speaker identifier from RTTM (e.g., SPEAKER_00)
- `turn_start` - Turn start time in seconds
- `turn_end` - Turn end time in seconds
- `turn_duration` - Turn duration in seconds

**Feature Columns:**
- 103 acoustic features (see Features section above)
- All numeric values rounded to 3 decimal places

### Example Output

```csv
speaker,turn_start,turn_end,turn_duration,spectral_centroid_mean,spectral_centroid_std,...
SPEAKER_00,0.000,2.350,2.350,1523.456,234.567,...
SPEAKER_01,2.450,4.270,1.820,1678.234,198.432,...
SPEAKER_00,4.500,6.890,2.390,1545.123,221.098,...
```

## Processing Details

### Audio Preprocessing
- Stereo audio is automatically converted to mono (channel averaging)
- Original sample rate is preserved
- No additional filtering or normalization applied

### Turn Filtering
- **Minimum turn duration: 0.1 seconds**
- Turns shorter than 100ms are skipped
- This prevents extraction errors on very brief segments

### Feature Extraction
- Features extracted on entire turn segment (not frame-by-frame then aggregated)
- Librosa default parameters used for all features
- NaN values may appear if extraction fails for specific features

### Error Handling
- Files without matching RTTM files are skipped
- Individual turn failures are logged but don't stop processing
- Files with zero valid turns will not produce output CSVs

## Logging

The script provides detailed console output:
- Script startup confirmation
- Input/output paths
- Per-file processing status
- Turn counts and skipped turns
- Error messages with stack traces
- Summary statistics

Example output:
```
================================================================================
LIBROSA PER-TURN EXTRACTOR
================================================================================
Input:  /data/audio
Output: /data/features

[1/50]
Processing: participant001_task
  Found 45 speaker turns
  ✓ Saved 43 turns (103 features)

[2/50]
Processing: participant002_task
  Found 38 speaker turns
  ✓ Saved 36 turns (103 features)
...
================================================================================
✅ Complete! Processed 48/50 files (2 skipped)
📁 Output: /data/features
================================================================================
```

## Comparison with Parselmouth Features

| Feature Category | Librosa (103) | Parselmouth (25) | Overlap |
|-----------------|---------------|------------------|---------|
| Spectral | 48 | 4 | Partial (different algorithms) |
| MFCCs | 26 | 0 | None |
| Chroma | 24 | 0 | None |
| Energy | 4 | 2 | Conceptual (RMS vs. Intensity) |
| Voice Quality | 0 | 4 | None (HNR, CPP, Jitter, Shimmer) |
| Formants | 0 | 8 | None |
| Prosody | 0 | 5 | None (speaking rate, pauses) |
| Pitch | 0 | 2 | None (F0 statistics) |
| Custom | 1 (HF500) | 0 | None |

**Key Differences:**
- **Librosa**: Focuses on spectral, timbral, and harmonic content
- **Parselmouth**: Focuses on voice quality, prosody, and formant analysis
- **Complementary**: Different acoustic perspectives of the same speech

## Assumptions and Limitations

### Assumptions
- RTTM files are accurate and complete
- Audio files are valid and readable by soundfile
- Speaker turns are correctly identified and labeled
- Minimum turn duration of 0.1s contains sufficient signal

### Limitations
- No pitch (F0) extraction - use Parselmouth for pitch features
- No formant tracking - use Parselmouth for formant features
- No voice quality measures (jitter, shimmer, HNR, CPP)
- No prosodic features (speaking rate, pauses, articulation rate)
- MFCCs may be sensitive to recording conditions
- Chroma features designed for music; applicability to speech varies
- HF500 is a simple custom feature, not validated clinically

### When to Use Librosa vs. Parselmouth
**Use Librosa for:**
- Spectral and timbral analysis
- Machine learning with MFCCs
- Harmonic content (chroma)
- General acoustic characterization

**Use Parselmouth for:**
- Voice quality assessment
- Prosodic analysis
- Formant tracking
- Clinical voice evaluation
- Speaking rate and timing analysis

**Use Both for:**
- Comprehensive acoustic profiling
- Multi-method validation
- Comparing different feature extraction approaches (ASCEND methodology)

## Troubleshooting

### "No RTTM file" warnings
- Ensure RTTM files have same base name as WAV files
- Check file extensions (.rttm, not .RTTM or .txt)

### "No valid turns extracted"
- Check if all turns are < 0.1 seconds (too short)
- Verify RTTM file is properly formatted
- Check audio file is not corrupted

### "Feature extraction warning"
- Usually occurs on very short or silent segments
- Turn will be skipped, processing continues
- Check audio quality if this happens frequently

### Memory issues with large files
- Librosa loads entire audio file into memory
- For very long recordings, consider splitting audio first
- Monitor system memory usage during processing

### Slow processing
- Librosa spectral features are computationally intensive
- STFT computation (for HF500) adds overhead
- Processing time scales with audio duration and sample rate
- Expected: ~2-5 seconds per turn depending on duration

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

## Version History

- **v1.0** (2025-10-21): Initial release
  - 103 librosa-based features
  - Per-turn extraction from RTTM diarization
  - Consistent with Parselmouth script structure

## Support

For issues or questions:
1. Check RTTM file format and audio file integrity
2. Verify all dependencies are correctly installed
3. Review log output for specific error messages
4. Ensure minimum turn duration threshold is appropriate for your data
