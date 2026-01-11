# Parselmouth 0.1-Second Window Feature Extractor

## Overview
Extracts acoustic features from audio files using 0.1-second sliding windows. Each window is analyzed independently to provide temporal dynamics of speech features.

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

## Usage

```bash
python3 parselmouth_standalone.py --input /path/to/wav/files --output /path/to/output/dir
```

### Arguments
- `--input` (required): Directory containing WAV files to process
- `--output` (required): Directory where CSV output files will be saved

### Example
```bash
python3 parselmouth_standalone.py \
    --input ~/audio/recordings \
    --output ~/audio/features
```

## Input Requirements

### Audio Files
- **Format**: WAV (`.wav` extension)
- **Channels**: Mono or stereo (stereo will be converted to mono by averaging)
- **Sample Rate**: Any (VoiceSpeechHealth handles resampling)
- **Duration**: Minimum 0.1 seconds

### File Organization
```
input_directory/
├── recording1.wav
├── recording2.wav
└── recording3.wav
```

## Output Format

### Files Generated
For each input WAV file, creates:
```
{filename}_parselmouth_0.1sec.csv
```

### CSV Structure

**Metadata Columns (4):**
- `window_start` - Start time of window in seconds
- `window_center` - Center time of window in seconds  
- `window_end` - End time of window in seconds
- `window_duration` - Duration of window (always 0.1)

**Feature Columns (25):**

#### Speech Rate Features (5) - **May be NaN**
- `Speaking_Rate` - Syllables per second
- `Articulation_Rate` - Syllables per second (excluding pauses)
- `Phonation_Ratio` - Proportion of time spent speaking
- `Pause_Rate` - Number of pauses per second
- `Mean_Pause_Duration` - Average pause length in seconds

**Note:** Speech rate features require longer windows (1+ seconds) to detect pauses and count syllables. In 0.1-second windows, these values will typically be NaN or unreliable.

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
- `mean_F1_Loc` - Mean first formant frequency (Hz)
- `std_F1_Loc` - Standard deviation of F1 (Hz)
- `mean_B1_Loc` - Mean first formant bandwidth (Hz)
- `std_B1_Loc` - Standard deviation of B1 (Hz)
- `mean_F2_Loc` - Mean second formant frequency (Hz)
- `std_F2_Loc` - Standard deviation of F2 (Hz)
- `mean_B2_Loc` - Mean second formant bandwidth (Hz)
- `std_B2_Loc` - Standard deviation of B2 (Hz)

#### Spectral Moment Features (4)
- `Spectral_Gravity` - Center of gravity of spectrum
- `Spectral_Std_Dev` - Standard deviation of spectrum
- `Spectral_Skewness` - Skewness of spectrum
- `Spectral_Kurtosis` - Kurtosis of spectrum

### Example Output
```csv
window_start,window_center,window_end,window_duration,Speaking_Rate,Articulation_Rate,...
0.0,0.05,0.1,0.1,,,,,215.3,2.1,67.4,1.2,12.5,...
0.1,0.15,0.2,0.1,,,,,218.7,2.3,68.1,1.1,13.2,...
0.2,0.25,0.3,0.1,,,,,220.1,2.0,67.9,1.3,12.8,...
```

## Assumptions and Limitations

### Window Size
- **Fixed at 0.1 seconds** (100 milliseconds)
- Non-overlapping windows
- Final partial window (< 0.1s) is discarded

### Speech Rate Features
- **Will be NaN or unreliable** in 0.1-second windows
- Requires ~1+ seconds to detect pauses and syllables
- Use longer windows or aggregate post-hoc if these features are needed

### Audio Processing
- Stereo files are converted to mono by averaging channels
- Very short audio segments (< 0.1s) may fail extraction
- Temporary WAV files created in `/tmp/` during processing

### Performance
- **Processing speed**: ~10 windows per second
- **Memory usage**: ~200-500 MB depending on file size
- **90-second file**: ~15-20 seconds processing time
- **5-minute file**: ~1-2 minutes processing time

## Troubleshooting

### No output produced
- Check that input directory contains `.wav` files (lowercase extension)
- Verify output directory is writable
- Check console output for errors

### Missing speech rate features
- **Expected behavior** - these features cannot be calculated on 0.1s windows
- See "Speech Rate Features" section above

### Import errors
```bash
# Verify VoiceSpeechHealth installation
python3 -c "import voicespeechhealth; print('OK')"

# If fails, reinstall:
pip install git+https://github.com/nickcummins41/VoiceSpeechHealth.git
```

### Memory issues with large files
- Process files individually rather than batch
- Close other memory-intensive applications

## Technical Details

### Feature Extraction Parameters
- `pitch_floor`: Auto-detected per file (typically ~75 Hz)
- `pitch_ceiling`: Auto-detected per file (typically ~500 Hz)
- `frame_shift`: 0.005 seconds (5 ms) for time-series features
- `window_size`: 0.025 seconds (25 ms) for spectral moments

### Extraction Process
1. Load full audio file
2. Split into non-overlapping 0.1-second windows
3. For each window:
   - Save temporary WAV file
   - Extract pitch range with `extract_pitch_values()`
   - Extract all 25 features
   - Store results
4. Combine all windows into DataFrame
5. Save CSV with metadata columns first

## Use Cases

### Appropriate Use
- **Temporal dynamics analysis** - track how features change over time
- **Event detection** - identify specific acoustic events
- **Time-series modeling** - features as continuous signals
- **Prosodic analysis** - intonation and intensity patterns over time

### Inappropriate Use
- **Speech rate analysis** - use longer windows (≥1 second)
- **Global speaker characteristics** - use aggregated/full-file analysis
- **Pause detection** - requires longer context

## Citation

If using VoiceSpeechHealth package:
```
Cummins, N. (2024). VoiceSpeechHealth: Acoustic feature extraction toolkit.
GitHub: https://github.com/nickcummins41/VoiceSpeechHealth
```

## License
Follows VoiceSpeechHealth package licensing.

## Version
- **Script version**: 1.0
- **Last updated**: October 2025
