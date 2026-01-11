# McAdams Voice Anonymization Scripts Documentation

## Overview

These Python scripts implement the McAdams coefficient method for voice anonymization, originally developed by the Voice Privacy Challenge 2022. The method modifies formant frequencies while preserving speech intelligibility, making it suitable for de-identifying pathological voice recordings.

## Scripts

### 1. Path_McAdams_Max.py
**Optimized for M3 Max with Multiprocessing**

#### Features
- Parallel processing using 10 processes (M3 Max performance cores)
- Recursive directory processing
- Maintains nested directory structure
- Progress tracking with tqdm
- Error handling and reporting

#### Configuration
```python
input_dir = "/Volumes/video_research/ForVOX/DeID(McAdams)"
output_dir = "/Volumes/video_research/ForVOX/DeID_McAdams_Anonymized"
mcadams_coef = 0.8
num_processes = 10
```

#### Output Format
- Preserves original directory structure
- Adds `anonymized_` prefix to filenames
- Example: `audio.wav` → `anonymized_audio.wav`

---

### 2. Path_McAdams_nest.py
**Single-threaded Sequential Processing**

#### Features
- Sequential processing with progress bar
- Recursive directory traversal
- Simpler implementation for smaller datasets

#### Configuration
```python
input_dir = "/Volumes/Databackup2025/mybox-selected/CSAND/CSA/SplicedAudio"
output_dir = "/Users/peterpressman/Desktop/Path_McAdams_for_InPlace_Deidentified"
mcadams_coef = 0.8
```

#### Output Format
- Preserves directory structure
- Adds `anon_` prefix to filenames

---

## McAdams Method Explained

### Algorithm Overview
The McAdams coefficient method modifies the vocal tract characteristics while preserving the excitation signal:

1. **Frame-based Processing**
   - Window length: 20ms
   - Shift length: 10ms (50% overlap)
   - Hanning window for analysis/synthesis

2. **Linear Predictive Coding (LPC)**
   - Order: 20 coefficients
   - Extracts vocal tract filter characteristics
   - Computes poles representing formant frequencies

3. **Pole Modification**
   - Modifies pole angles using: `new_angle = angle^mcadams`
   - Preserves pole magnitudes
   - Maintains conjugate pairs for stability

4. **Reconstruction**
   - Creates new LPC filter from modified poles
   - Applies to residual excitation
   - Overlap-add synthesis for output

### McAdams Coefficient Values

- **< 1.0**: Expands formant spacing (higher angles), contracts for lower angles
- **= 1.0**: No modification (identity transform)
- **> 1.0**: Contracts formant spacing (higher angles), expands for lower angles
- **0.8** (default): Moderate anonymization with good intelligibility

---

## Dependencies

```python
numpy
scipy
librosa
soundfile
tqdm
multiprocessing  # Path_McAdams_Max.py only
```

### Installation
```bash
pip install numpy scipy librosa soundfile tqdm
```

---

## Usage Examples

### Running the Multiprocessing Version
```bash
python3 Path_McAdams_Max.py
```

### Running the Sequential Version
```bash
python3 Path_McAdams_nest.py
```

### Customizing Parameters

Modify the McAdams coefficient for different anonymization levels:

```python
# Stronger anonymization (more modification)
anonymizer.process_directory(mcadams_coef=0.6)

# Weaker anonymization (less modification)
anonymizer.process_directory(mcadams_coef=0.9)
```

---

## File Processing

### Input Requirements
- WAV audio files
- Mono or stereo (automatically converted to mono)
- Any sample rate supported by soundfile

### Output Characteristics
- 32-bit float WAV format
- Normalized to prevent clipping
- Same sample rate as input
- Mono channel

---

## Performance Comparison

| Script | Processing Method | Speed | Best For |
|--------|------------------|-------|----------|
| Path_McAdams_Max.py | 10 parallel processes | ~10x faster | Large datasets, M3 Max |
| Path_McAdams_nest.py | Sequential | Baseline | Small datasets, any CPU |

---

## Directory Structure Example

### Input
```
/input_dir/
├── participant1/
│   ├── recording1.wav
│   └── recording2.wav
└── participant2/
    └── recording3.wav
```

### Output (Max version)
```
/output_dir/
├── participant1/
│   ├── anonymized_recording1.wav
│   └── anonymized_recording2.wav
└── participant2/
    └── anonymized_recording3.wav
```

### Output (Nest version)
```
/output_dir/
├── participant1/
│   ├── anon_recording1.wav
│   └── anon_recording2.wav
└── participant2/
    └── anon_recording3.wav
```

---

## Error Handling

Both scripts include error handling:
- Skips files that fail to process
- Reports errors with file paths
- Continues processing remaining files
- Provides summary of successful/failed files (Max version)

---

## Credits

**Original Algorithm:**
- Jose Patino, Massimiliano Todisco, Pramod Bachhav, Nicholas Evans
- Audio Security and Privacy Group, EURECOM
- Voice Privacy Challenge 2022

**References:**
- [Voice Privacy Challenge 2022](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2022)
- [Anonymized Speech Diagnostics](https://github.com/zhu00121/Anonymized-speech-diagnostics)

**Modified by:**
- Soroosh Tayebi Arasteh (FAU)

---

## Technical Notes

### Memory Considerations
- Each process loads one file at a time
- Memory usage scales with: (file_size × num_processes)
- For large files, consider reducing `num_processes`

### CPU Optimization
- M3 Max: 10 performance cores (Path_McAdams_Max.py default)
- Adjust `num_processes` based on your CPU
- `cpu_count()` available but not used by default

### Audio Quality
- Maintains speech intelligibility
- Preserves prosody and timing
- Modifies speaker identity characteristics
- Suitable for pathological voice analysis

---

## Troubleshooting

### No files found
- Verify input directory path
- Check file extensions (must be `.wav`)
- Ensure proper mount of network drives

### Out of memory
- Reduce `num_processes` in Path_McAdams_Max.py
- Process smaller batches
- Check available RAM

### Slow processing
- Use Path_McAdams_Max.py for large datasets
- Ensure input/output on fast storage (not network drives)
- Verify CPU isn't throttled

---

## License

This implementation uses the McAdams method from the Voice Privacy Challenge, which is available under open-source terms. Refer to the original repositories for specific licensing information.