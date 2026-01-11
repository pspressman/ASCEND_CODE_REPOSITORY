# Parselmouth Per-Speaker Aggregated Feature Extractor

## Overview
Extracts acoustic features aggregated across all turns for each speaker in conversational audio. Produces one row per speaker per file, with features calculated over all concatenated speech segments for that speaker.

## Dependencies

### Python Version
- **Python 3.8+** (tested with 3.9, 3.10, 3.11)

### Required Packages
```bash
pip install voicespeechhealth
pip install soundfile
pip install pandas
pip install parselmouth
pip install numpy
```

**VoiceSpeechHealth Installation:**
```bash
pip install git+https://github.com/nickcummins41/VoiceSpeechHealth.git
```

### Additional Requirements
- **Pyannote diarization files** (`.rttm` format)
- RTTM files must have same base name as WAV files

## Usage

```bash
python3 parselmouth_per_speaker_aggregated.py --input /path/to/wav/files --output /path/to/output.csv
```

### Arguments
- `--input` (required): Directory containing WAV and RTTM files
- `--output` (required): Single CSV file path for all results

### Example
```bash
python3 parselmouth_per_speaker_aggregated.py \
    --input ~/conversations/audio \
    --output ~/conversations/all_speakers_features.csv
```

## Input Requirements

### Required Files
For each audio file, TWO files are required:
1. **WAV file**: `recording.wav`
2. **RTTM file**: `recording.rttm` (same base name)

### File Organization
```
input_directory/
├── conversation1.wav
├── conversation1.rttm
├── conversation2.wav
├── conversation2.rttm
├── conversation3.wav
└── conversation3.rttm
```

### Audio Format
- **Format**: WAV (`.wav` extension)
- **Channels**: Mono or stereo (stereo → mono by averaging)
- **Sample Rate**: Any (VoiceSpeechHealth handles resampling)
- **Duration**: Any

### RTTM Format
Standard pyannote diarization format:
```
SPEAKER filename 1 0.000 2.350 <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER filename 1 2.450 1.820 <NA> <NA> SPEAKER_01 <NA> <NA>
SPEAKER filename 1 4.350 3.120 <NA> <NA> SPEAKER_00 <NA> <NA>
```

**RTTM Fields:**
1. Label: `SPEAKER`
2. Filename
3. Channel
4. Start time (seconds)
5. Duration (seconds)
6-7. `<NA>` placeholders
8. Speaker ID: `SPEAKER_00`, `SPEAKER_01`, etc.
9-10. `<NA>` placeholders

## Output Format

### Single CSV File
All results from all files combined into ONE CSV:
```
all_speakers_features.csv
```

**Row Structure:**
- 1 file with 1 speaker → 1 row
- 1 file with 2 speakers → 2 rows
- 1 file with 3 speakers → 3 rows
- 10 files with 2 speakers each → 20 rows total

### CSV Structure

**Metadata Columns (4):**
- `filename` - Source WAV file (without extension)
- `speaker` - Speaker ID from RTTM (e.g., SPEAKER_00, SPEAKER_01)
- `total_duration` - Total seconds of speech for this speaker (all turns combined)
- `num_turns` - Number of speaking turns concatenated for this speaker

**Feature Columns (25):**

#### Speech Rate Features (5)
- `Speaking_Rate` - Syllables per second (aggregated across all turns)
- `Articulation_Rate` - Syllables per second excluding pauses
- `Phonation_Ratio` - Overall proportion of time spent speaking
- `Pause_Rate` - Pauses per second (across all speech)
- `Mean_Pause_Duration` - Average pause length in seconds

**Note:** Most reliable speech rate estimates (long context from all turns combined).

#### Pitch Features (2)
- `mean_F0` - Mean fundamental frequency across all turns (Hz)
- `stdev_F0_Semitone` - Overall F0 variability in semitones

#### Intensity Features (2)
- `mean_dB` - Mean intensity across all turns (decibels)
- `range_ratio_dB` - Overall intensity range (decibels)

#### Voice Quality Features (4)
- `HNR_dB` - Overall Harmonics-to-Noise Ratio (decibels)
- `Spectral_Slope` - Aggregated spectral slope
- `Spectral_Tilt` - Aggregated spectral tilt
- `Cepstral_Peak_Prominence` - Mean CPP across all speech

#### Formant Features (8)
- `mean_F1_Loc`, `std_F1_Loc` - First formant (Hz)
- `mean_B1_Loc`, `std_B1_Loc` - First formant bandwidth (Hz)
- `mean_F2_Loc`, `std_F2_Loc` - Second formant (Hz)
- `mean_B2_Loc`, `std_B2_Loc` - Second formant bandwidth (Hz)

#### Spectral Moment Features (4)
- `Spectral_Gravity` - Overall spectral center of gravity
- `Spectral_Std_Dev` - Overall spectral standard deviation
- `Spectral_Skewness` - Overall spectral skewness
- `Spectral_Kurtosis` - Overall spectral kurtosis

### Example Output
```csv
filename,speaker,total_duration,num_turns,Speaking_Rate,Articulation_Rate,mean_F0,...
conversation1,SPEAKER_00,45.3,15,4.3,4.9,187.2,2.1,67.8,1.2,12.8,...
conversation1,SPEAKER_01,38.7,12,4.1,4.7,215.6,1.9,66.1,1.3,13.5,...
conversation2,SPEAKER_00,52.1,18,4.5,5.0,182.3,2.0,68.2,1.1,12.3,...
conversation2,SPEAKER_01,41.2,14,4.0,4.6,218.9,2.1,65.8,1.4,14.1,...
conversation3,SPEAKER_00,67.8,22,4.4,4.8,190.1,2.2,67.5,1.2,13.0,...
```

## Assumptions and Limitations

### Speaker Aggregation
- **All turns concatenated** - speaker's turns are joined in temporal order
- **No turn boundaries** - features calculated as if continuous speech
- **Gaps removed** - pauses between turns are not included in concatenated audio

### Minimum Duration
- **Minimum total speech**: 0.5 seconds per speaker
- Speakers with < 0.5s total speech are skipped with warning
- Based on sum of all turn durations

### Speaker Identification
- Uses speaker IDs exactly as provided in RTTM
- No speaker verification or validation
- Any number of speakers supported per file

### Diarization Quality
- **Assumes accurate diarization** from pyannote
- Diarization errors will affect which speech is assigned to each speaker
- Overlapping speech assigned to single speaker (per RTTM)

### Audio Processing
- Stereo audio converted to mono by averaging channels
- All speaker turns concatenated into single audio stream
- Temporary WAV files created in `/tmp/`
- Original turn order preserved in concatenation

### Performance
- **Processing speed**: ~2-5 seconds per speaker
- **10 files, 2 speakers each**: ~1-2 minutes total
- **Memory**: ~200-500 MB typical
- Most efficient of the three extraction scripts

## Troubleshooting

### No output produced
- Verify RTTM files exist with same base name as WAV
- Check RTTM format (must start with `SPEAKER`)
- Ensure at least one speaker meets minimum duration (0.5s)

### Speakers skipped
```
WARNING: Speaker SPEAKER_00 total audio too short, skipping
```
**Cause:** Speaker's total speech < 0.5 seconds  
**Solution:** Either:
- Lower threshold in code (line 164: `if len(speaker_audio) < sr * 0.5`)
- Accept that minimal speech cannot be reliably characterized

### Missing RTTM warnings
```
WARNING: conversation1.wav: No RTTM file, skipping
```
**Solution:** Generate RTTM files using pyannote or ensure they're in input directory

### Import errors
```bash
# Verify all dependencies
python3 -c "import voicespeechhealth, numpy, pandas, soundfile; print('OK')"
```

## Technical Details

### Feature Extraction Parameters
- `pitch_floor`: Auto-detected per speaker (typically ~75 Hz)
- `pitch_ceiling`: Auto-detected per speaker (typically ~500 Hz)
- `frame_shift`: 0.005 seconds (5 ms)
- `window_size`: 0.025 seconds (25 ms) for spectral moments
- `min_total_duration`: 0.5 seconds per speaker

### Extraction Process
1. Parse RTTM file to identify speakers and their turns
2. Group turns by speaker ID
3. Load full audio file
4. For each speaker:
   - Extract all turn segments
   - Concatenate segments using `numpy.concatenate()`
   - Skip if total duration < 0.5s
   - Save concatenated audio to temporary WAV
   - Extract pitch range
   - Extract all 25 features
   - Store with file and speaker metadata
5. Combine all results from all files
6. Save single CSV with all speakers

### Concatenation Order
Turns are concatenated in **temporal order** (as they appear in conversation):
```
Speaker A: [Turn 1] -------- [Turn 3] -------- [Turn 5]
           concatenated → [Turn1][Turn3][Turn5]
```

## Use Cases

### Appropriate Use
- **Speaker profiling** - characteristic acoustic features per speaker
- **Clinical assessment** - overall speech production metrics
- **Speaker comparison** - acoustic differences between participants
- **Diagnostic classification** - speaker-level predictions
- **Longitudinal studies** - tracking individual speakers over time
- **Inter-speaker analysis** - comparing speakers within conversations

### Comparison with Other Scripts
| Feature | 0.1s Windows | Per-Turn | **Per-Speaker Aggregated** |
|---------|--------------|----------|---------------------------|
| Output | Many CSVs | Many CSVs | **One CSV** |
| Rows/file | ~600/minute | ~50-100 | **1 per speaker** |
| Speech rate | Unreliable | Reliable | **Most reliable** |
| Temporal info | Finest | Turn-level | **None** |
| Turn boundaries | No | Yes | **No** |
| Best for | Time series | Conversations | **Speaker profiles** |
| Analysis level | Frame | Turn | **Speaker** |

### When to Use This Script
Use **per-speaker aggregated** when:
- Interested in overall speaker characteristics
- Need most reliable speech rate estimates
- Want simple speaker-level comparisons
- Planning machine learning with speaker as unit of analysis
- Need efficient processing (fastest of three scripts)

Use **per-turn** when:
- Need turn-level dynamics
- Studying conversational patterns
- Require turn boundary information

Use **0.1s windows** when:
- Need finest temporal resolution
- Time-series analysis required
- Studying prosodic contours

## Example Workflow

### Complete Analysis Pipeline
```bash
# 1. Generate diarization (if needed)
python3 generate_rttm.py --input audio/ --output audio/

# 2. Extract per-speaker features
python3 parselmouth_per_speaker_aggregated.py \
    --input audio/ \
    --output results/speaker_features.csv

# 3. Analyze results
python3 analyze_speakers.py --input results/speaker_features.csv
```

### Batch Processing Multiple Datasets
```bash
# Process multiple conversational datasets
for dataset in dataset1 dataset2 dataset3; do
    python3 parselmouth_per_speaker_aggregated.py \
        --input ~/data/$dataset/audio \
        --output ~/results/${dataset}_speakers.csv
done

# Combine all results
cat ~/results/*_speakers.csv > ~/results/all_datasets.csv
```

## RTTM Generation

If you need to generate RTTM files:

```python
from pyannote.audio import Pipeline

# Load pretrained pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

# Process all audio files
import glob
for audio_file in glob.glob("*.wav"):
    diarization = pipeline(audio_file)
    
    # Save RTTM
    rttm_file = audio_file.replace(".wav", ".rttm")
    with open(rttm_file, "w") as f:
        diarization.write_rttm(f)
```

## Citation

If using VoiceSpeechHealth:
```
Cummins, N. (2024). VoiceSpeechHealth: Acoustic feature extraction toolkit.
GitHub: https://github.com/nickcummins41/VoiceSpeechHealth
```

If using pyannote diarization:
```
Bredin, H., et al. (2020). pyannote.audio: neural building blocks for 
speaker diarization. ICASSP 2020.
```

## License
Follows VoiceSpeechHealth and pyannote licensing.

## Version
- **Script version**: 1.0
- **Last updated**: October 2025
