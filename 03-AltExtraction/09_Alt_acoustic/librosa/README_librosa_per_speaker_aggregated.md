# Librosa Per-Speaker Aggregated Feature Extractor

## Overview

This script extracts 103 librosa-based acoustic features aggregated across all turns for each speaker in each audio file. It processes audio files alongside their RTTM diarization files and produces a single CSV containing one row per speaker per file.

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
python3 librosa_per_speaker_aggregated.py --input /path/to/wavs --output /path/to/output.csv
```

### Arguments

- `--input` (required): Directory containing WAV files and their corresponding RTTM files
- `--output` (required): Path to output CSV file (will be created, including parent directories)

### Example

```bash
python3 librosa_per_speaker_aggregated.py \
    --input /data/ascend/audio \
    --output /data/ascend/features/librosa_aggregated.csv
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
SPEAKER filename 1 4.500 2.390 <NA> <NA> SPEAKER_00 <NA> <NA>
```

Fields: `SPEAKER filename channel start_time duration _ _ speaker_id _ _`

## Output Format

### File Naming

Single CSV file specified by `--output` argument.

### CSV Structure

**Metadata Columns:**
- `filename` - Base name of audio file (without extension)
- `speaker` - Speaker identifier from RTTM (e.g., SPEAKER_00)
- `total_duration` - Total duration of all speaker's turns in seconds
- `num_turns` - Number of turns by this speaker

**Feature Columns:**
- 103 acoustic features (see Features section above)
- All numeric values rounded to 3 decimal places

### Example Output

```csv
filename,speaker,total_duration,num_turns,spectral_centroid_mean,spectral_centroid_std,...
participant001_task,SPEAKER_00,45.234,12,1534.456,245.678,...
participant001_task,SPEAKER_01,38.567,10,1689.123,212.345,...
participant002_task,SPEAKER_00,52.891,15,1512.789,238.901,...
participant002_task,SPEAKER_01,41.234,11,1701.456,198.765,...
```

## Processing Details

### Feature Aggregation Method

**Per-Speaker Concatenation:**
1. All turns for each speaker are identified from RTTM
2. Audio segments for all turns are concatenated chronologically
3. Features are extracted from the concatenated audio segment
4. This provides speaker-level acoustic profile across entire recording

**Alternative (not implemented):**
- Could extract per-turn then average features
- Current approach captures speaker consistency across longer audio

### Audio Preprocessing
- Stereo audio is automatically converted to mono (channel averaging)
- Original sample rate is preserved
- No additional filtering or normalization applied

### Speaker Filtering
- **Minimum total duration: 0.5 seconds**
- Speakers with less than 500ms total speaking time are skipped
- This ensures sufficient audio for reliable feature extraction

### Feature Extraction
- Features extracted on concatenated speaker audio (all turns combined)
- Librosa default parameters used for all features
- NaN values may appear if extraction fails for specific features

### Error Handling
- Files without matching RTTM files are skipped
- Individual speaker failures are logged but don't stop processing
- Files with zero valid speakers will not contribute to output

## Logging

The script provides detailed console output:
- Script startup confirmation
- Input/output paths
- Per-file processing status
- Speaker counts and turn counts
- Error messages with stack traces
- Summary statistics

Example output:
```
================================================================================
LIBROSA PER-SPEAKER AGGREGATED EXTRACTOR
================================================================================
Input:  /data/audio
Output: /data/features/librosa_aggregated.csv

[1/50]
Processing: participant001_task
  Found 2 speakers
  Speaker SPEAKER_00: 12 turns
  Speaker SPEAKER_01: 10 turns

[2/50]
Processing: participant002_task
  Found 2 speakers
  Speaker SPEAKER_00: 15 turns
  Speaker SPEAKER_01: 11 turns
...
================================================================================
✅ Complete! Processed 48/50 files (2 skipped)
📊 Total speakers extracted: 96
📁 Output: /data/features/librosa_aggregated.csv
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

## Per-Turn vs. Per-Speaker Aggregated

| Aspect | Per-Turn | Per-Speaker Aggregated |
|--------|----------|------------------------|
| **Output** | One CSV per audio file | Single CSV for all files |
| **Rows** | One per speaker turn | One per speaker per file |
| **Use Case** | Turn-by-turn dynamics, conversational analysis | Speaker-level profiles, participant comparison |
| **Temporal Info** | Preserves turn timestamps | Loses turn boundaries |
| **Data Size** | Larger (more rows) | Smaller (aggregated) |
| **ML Use** | Sequence models, temporal patterns | Traditional ML, speaker classification |

**When to use Per-Speaker Aggregated:**
- Comparing speakers across recordings
- Training speaker-level classifiers
- Identifying speaker characteristics independent of conversation dynamics
- Reducing data dimensionality
- ASCEND cross-method validation (aggregated features comparable across toolkits)

**When to use Per-Turn:**
- Analyzing conversational dynamics
- Prosodic accommodation studies
- Turn-taking patterns
- Temporal progression of features within recording

## Assumptions and Limitations

### Assumptions
- RTTM files are accurate and complete
- Audio files are valid and readable by soundfile
- Speaker turns are correctly identified and labeled
- Concatenated speaker audio (0.5s minimum) contains sufficient signal
- Speaker characteristics are relatively stable across turns

### Limitations
- No pitch (F0) extraction - use Parselmouth for pitch features
- No formant tracking - use Parselmouth for formant features
- No voice quality measures (jitter, shimmer, HNR, CPP)
- No prosodic features (speaking rate, pauses, articulation rate)
- MFCCs may be sensitive to recording conditions
- Chroma features designed for music; applicability to speech varies
- HF500 is a simple custom feature, not validated clinically
- Concatenation may create artifacts at turn boundaries
- Turn-to-turn variation is lost in aggregation

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

### "Speaker total audio too short" warnings
- Speaker has less than 0.5 seconds total speaking time
- Consider lowering threshold if analyzing very brief recordings
- This is normal for brief participant responses

### "No results to save" error
- All speakers were below minimum duration threshold
- No valid RTTM files found
- Check RTTM format and audio file integrity

### Memory issues with large datasets
- Processing hundreds of files creates large final CSV
- Consider processing in batches if memory constrained
- Each speaker requires loading full audio file into memory

### Slow processing
- Librosa spectral features are computationally intensive
- STFT computation (for HF500) adds overhead
- Processing time scales with total audio duration
- Expected: ~5-10 seconds per speaker depending on total duration

## ASCEND Methodology Context

This script is part of the ASCEND (Automated Speech Comparison Engine for Neurocognitive Detection) project, which systematically compares different feature extraction methods for speech-based dementia detection.

**Purpose in ASCEND:**
- Third acoustic feature toolkit (alongside eGeMAPS and Parselmouth)
- Tests whether spectral/timbral features discriminate disease
- Validates MFCC-based approaches common in machine learning
- Provides complementary feature space to voice quality measures

**Expected Use:**
1. Extract features using all three toolkits (eGeMAPS, Parselmouth, Librosa)
2. Train separate classifiers on each feature set
3. Compare diagnostic performance across methods
4. Identify most robust features for clinical deployment

**Integration with Other ASCEND Components:**
- Use same RTTM diarization as Parselmouth scripts
- Compatible with multi-ASR pipeline (WhisperX, Faster-Whisper, Vosk)
- Supports de-identification validation workflow
- Enables equipment comparison (studio vs. consumer microphones)

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
  - Per-speaker aggregated extraction from RTTM diarization
  - Single output CSV for all files
  - Consistent with Parselmouth script structure

## Support

For issues or questions:
1. Check RTTM file format and audio file integrity
2. Verify all dependencies are correctly installed
3. Review log output for specific error messages
4. Ensure minimum speaker duration threshold is appropriate for your data
5. Confirm output directory is writable
