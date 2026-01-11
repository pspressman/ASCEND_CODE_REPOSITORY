# Parselmouth Per-Turn Feature Extractor

## Overview
Extracts acoustic features for each speaker turn in conversational audio using pyannote diarization. Each turn is analyzed independently, preserving turn-level dynamics and speaker-specific patterns.

## Dependencies

### Python Version
- **Python 3.8+** (tested with 3.9, 3.10, 3.11)

### Required Packages
```bash
pip install voicespeechhealth
pip install soundfile
pip install pandas
pip install parselmouth
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
python3 parselmouth_per_turn.py --input /path/to/wav/files --output /path/to/output/dir
```

### Arguments
- `--input` (required): Directory containing WAV and RTTM files
- `--output` (required): Directory where per-file CSV outputs will be saved

### Example
```bash
python3 parselmouth_per_turn.py \
    --input ~/conversations/audio \
    --output ~/conversations/features
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
└── conversation3.wav
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
2. Filename: audio file name
3. Channel: typically `1`
4. Start time: seconds
5. Duration: seconds
6-7. `<NA>` placeholders
8. Speaker ID: `SPEAKER_00`, `SPEAKER_01`, etc.
9-10. `<NA>` placeholders

## Output Format

### Files Generated
For each input WAV file, creates:
```
{filename}_parselmouth_per_turn.csv
```

**Note:** Files without matching RTTM are skipped with a warning.

### CSV Structure

**Metadata Columns (4):**
- `speaker` - Speaker ID from RTTM (e.g., SPEAKER_00, SPEAKER_01)
- `turn_start` - Start time of turn in seconds
- `turn_end` - End time of turn in seconds
- `turn_duration` - Duration of turn in seconds

**Feature Columns (25):**

#### Speech Rate Features (5)
- `Speaking_Rate` - Syllables per second
- `Articulation_Rate` - Syllables per second (excluding pauses)
- `Phonation_Ratio` - Proportion of time spent speaking
- `Pause_Rate` - Number of pauses per second
- `Mean_Pause_Duration` - Average pause length in seconds

**Note:** More reliable than 0.1s windows, but may still be limited for very short turns.

#### Pitch Features (2)
- `mean_F0` - Mean fundamental frequency in Hz
- `stdev_F0_Semitone` - Standard deviation of F0 in semitones

#### Intensity Features (2)
- `mean_dB` - Mean intensity in decibels
- `range_ratio_dB` - Intensity range ratio in decibels

#### Voice Quality Features (4)
- `HNR_dB` - Harmonics-to-Noise Ratio in decibels
- `Spectral_Slope` - Spectral slope
- `Spectral_Tilt` - Spectral tilt
- `Cepstral_Peak_Prominence` - Mean CPP value

#### Formant Features (8)
- `mean_F1_Loc`, `std_F1_Loc` - First formant location and variability
- `mean_B1_Loc`, `std_B1_Loc` - First formant bandwidth and variability
- `mean_F2_Loc`, `std_F2_Loc` - Second formant location and variability
- `mean_B2_Loc`, `std_B2_Loc` - Second formant bandwidth and variability

#### Spectral Moment Features (4)
- `Spectral_Gravity` - Center of gravity of spectrum
- `Spectral_Std_Dev` - Standard deviation of spectrum
- `Spectral_Skewness` - Skewness of spectrum
- `Spectral_Kurtosis` - Kurtosis of spectrum

### Example Output
```csv
speaker,turn_start,turn_end,turn_duration,Speaking_Rate,Articulation_Rate,mean_F0,...
SPEAKER_00,0.000,2.350,2.350,4.2,4.8,185.3,2.1,67.4,1.2,12.5,...
SPEAKER_01,2.450,4.270,1.820,3.8,4.5,220.1,1.8,65.2,1.4,14.2,...
SPEAKER_00,4.350,7.470,3.120,4.5,5.1,188.7,2.0,68.1,1.1,13.0,...
SPEAKER_01,7.550,9.890,2.340,4.0,4.6,218.4,1.9,66.3,1.3,13.8,...
```

## Assumptions and Limitations

### Turn Length
- **Minimum turn duration**: 0.1 seconds
- Turns shorter than 0.1s are automatically skipped
- Very short turns may have unreliable features

### Speaker Identification
- Uses speaker IDs exactly as provided in RTTM
- No speaker verification or validation
- Multiple speakers per file supported (any number)

### Diarization Quality
- **Assumes accurate diarization** from pyannote
- Diarization errors will propagate to feature extraction
- Overlapping speech segments are assigned to one speaker (per RTTM)

### Audio Processing
- Each turn extracted and analyzed independently
- No context from surrounding turns
- Stereo audio averaged to mono
- Temporary WAV files created in `/tmp/`

### Performance
- **Processing speed**: ~0.5-1 second per turn
- **100-turn conversation**: ~1-2 minutes
- **Memory**: ~200-500 MB typical
- Faster than 0.1s windows due to fewer extractions

## Troubleshooting

### No output produced
- Verify RTTM files exist with same base name as WAV
- Check RTTM format (must start with `SPEAKER`)
- Ensure RTTM file is not empty

### Missing RTTM warnings
```
WARNING: conversation1.wav: No RTTM file, skipping
```
**Solution:** Generate RTTM files using pyannote or ensure they're in the input directory

### Too many turns skipped
```
Turn X too short (0.05s), skipping
```
**Cause:** Diarization produced very short turns (< 0.1s)  
**Solution:** Either:
- Adjust diarization parameters to merge short segments
- Accept that very short turns cannot be reliably analyzed

### Import errors
```bash
# Verify VoiceSpeechHealth
python3 -c "import voicespeechhealth; print('OK')"
```

## Technical Details

### Feature Extraction Parameters
- `pitch_floor`: Auto-detected per turn (typically ~75 Hz)
- `pitch_ceiling`: Auto-detected per turn (typically ~500 Hz)
- `frame_shift`: 0.005 seconds (5 ms)
- `window_size`: 0.025 seconds (25 ms) for spectral moments
- `min_turn_duration`: 0.1 seconds

### Extraction Process
1. Parse RTTM file to identify all turns
2. Load full audio file
3. For each turn:
   - Extract audio segment based on timestamps
   - Skip if < 0.1 seconds
   - Save temporary WAV
   - Extract pitch range
   - Extract all 25 features
   - Store with turn metadata
4. Combine all turns into DataFrame
5. Save CSV per file

## Use Cases

### Appropriate Use
- **Turn-taking analysis** - how features change across turns
- **Conversational dynamics** - speaker-specific patterns in dialogue
- **Prosodic accommodation** - pitch/intensity matching between speakers
- **Clinical assessments** - turn-level speech production analysis
- **Interaction analysis** - temporal structure of conversations

### Comparison with Other Scripts
| Feature | 0.1s Windows | Per-Turn | Per-Speaker Aggregated |
|---------|--------------|----------|------------------------|
| Rows per file | ~600/minute | ~50-100 | 1-3 |
| Speech rate | Unreliable | Reliable | Most reliable |
| Temporal dynamics | Finest | Turn-level | None |
| Turn boundaries | No | Yes | No |
| Best for | Time series | Conversations | Speaker profiles |

## RTTM Generation

If you need to generate RTTM files, use pyannote:

```python
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
diarization = pipeline("audio.wav")

# Save to RTTM
with open("audio.rttm", "w") as rttm:
    diarization.write_rttm(rttm)
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
