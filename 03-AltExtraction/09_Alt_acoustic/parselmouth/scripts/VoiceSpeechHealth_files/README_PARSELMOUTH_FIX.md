# Parselmouth 0.1-Second Feature Extractor
## Environment Fix and Usage Guide

## Problem Summary

**Issue**: Script runs but produces no output due to stacked conda environments interfering with ml_env.

**Root Cause**: 
- Conda base environment was active: `(ml_env) (base)`
- `which python3` showed conda's Python instead of ml_env's Python
- This caused silent failures despite VoiceSpeechHealth being installed correctly

## Solution

Use the provided launcher script that explicitly uses ml_env's Python, bypassing conda entirely.

---

## Quick Start (3 Steps)

### 1. Make scripts executable
```bash
cd ~/Desktop/AudioThreeTest/parseltest  # or wherever you saved the scripts
chmod +x diagnose_environment.sh
chmod +x run_parselmouth_extraction.sh
chmod +x parselmouth_standalone_improved.py
```

### 2. Diagnose your environment (optional but recommended)
```bash
./diagnose_environment.sh
```

This will show:
- Current shell environment variables
- Python paths
- VoiceSpeechHealth installation status
- Whether conda is interfering

### 3. Run the extraction
```bash
# Using default paths (test file location)
./run_parselmouth_extraction.sh

# OR with custom paths
./run_parselmouth_extraction.sh /path/to/input/wavs /path/to/output/csvs
```

---

## What Each File Does

### `diagnose_environment.sh`
- Checks your Python environment setup
- Verifies VoiceSpeechHealth installation
- Identifies conda interference
- Validates test file exists

### `run_parselmouth_extraction.sh`
- **Key feature**: Bypasses conda by explicitly using ml_env's Python
- Unsets conda environment variables
- Handles command-line arguments or uses defaults
- Creates output directory automatically
- Provides clear error messages

### `parselmouth_standalone_improved.py`
- Enhanced version of your original script
- Better logging (both console and file)
- Environment validation on startup
- Progress tracking every 100 windows
- More detailed error messages
- Creates `parselmouth_extraction.log` file

---

## Default Paths

If you run with no arguments, the script uses:

```
Input:  ~/Desktop/AudioThreeTest/parseltest/
Output: ~/Desktop/AudioThreeTest/parseltest/output/
```

---

## Expected Behavior

### Successful run will show:
```
======================================
PARSELMOUTH 0.1-SECOND EXTRACTOR
======================================

✅ Environment Check:
   Python: /Users/peterpressman/MyDevelopment/Environments/Me_Labeller_venv/ml_env/bin/python3
   Script: ./parselmouth_standalone_improved.py

📁 Using default paths:
   Input:  /Users/peterpressman/Desktop/AudioThreeTest/parseltest
   Output: /Users/peterpressman/Desktop/AudioThreeTest/parseltest/output

🚀 Starting extraction...
======================================

2025-10-21 10:30:00 - INFO - ================================================================================
2025-10-21 10:30:00 - INFO - PARSELMOUTH 0.1-SECOND EXTRACTOR
2025-10-21 10:30:00 - INFO - ================================================================================
2025-10-21 10:30:00 - INFO - Input:  /Users/peterpressman/Desktop/AudioThreeTest/parseltest
2025-10-21 10:30:00 - INFO - Output: /Users/peterpressman/Desktop/AudioThreeTest/parseltest/output
2025-10-21 10:30:00 - INFO - Found 1 WAV files
2025-10-21 10:30:00 - INFO -   - CSA_MAC_Audacity_2-003-8_20SEP2022_Paralang.wav
2025-10-21 10:30:00 - INFO - ================================================================================

2025-10-21 10:30:00 - INFO - [1/1]
2025-10-21 10:30:00 - INFO - Processing: CSA_MAC_Audacity_2-003-8_20SEP2022_Paralang
2025-10-21 10:30:01 - INFO -   Audio loaded: 2058750 samples at 22050 Hz
2025-10-21 10:30:01 - INFO -   Duration: 93.3s, ~933 windows expected
2025-10-21 10:30:15 - INFO -   Processed 100/933 windows...
2025-10-21 10:30:30 - INFO -   Processed 200/933 windows...
...
2025-10-21 10:32:00 - INFO -   ✓ Saved 933 windows (25 features)
2025-10-21 10:32:00 - INFO - ================================================================================
2025-10-21 10:32:00 - INFO - ✅ Complete! Processed 1 files
2025-10-21 10:32:00 - INFO - ⏱️  Elapsed time: 0:02:00
2025-10-21 10:32:00 - INFO - 📁 Output: /Users/peterpressman/Desktop/AudioThreeTest/parseltest/output
2025-10-21 10:32:00 - INFO - ================================================================================

======================================
✅ Extraction completed successfully!

📊 Results saved to:
   /Users/peterpressman/Desktop/AudioThreeTest/parseltest/output
======================================
```

---

## Output Files

For each input WAV file, you'll get:
```
{filename}_parselmouth_0.1sec.csv
```

### CSV Structure:
- **Metadata columns** (4): window_start, window_center, window_end, window_duration
- **Feature columns** (25):
  - Speech rate features (5): Speaking_Rate, Articulation_Rate, Phonation_Ratio, Pause_Rate, Mean_Pause_Duration
  - Pitch features (2): mean_F0, stdev_F0_Semitone
  - Intensity features (2): mean_dB, range_ratio_dB
  - Quality features (4): HNR_dB, Spectral_Slope, Spectral_Tilt, Cepstral_Peak_Prominence
  - Formant features (8): mean_F1_Loc, std_F1_Loc, mean_B1_Loc, std_B1_Loc, mean_F2_Loc, std_F2_Loc, mean_B2_Loc, std_B2_Loc
  - Spectral moments (4): Spectral_Gravity, Spectral_Std_Dev, Spectral_Skewness, Spectral_Kurtosis

---

## Troubleshooting

### Problem: "ml_env Python not found"
**Solution**: Check your ml_env path. Edit `run_parselmouth_extraction.sh` line 11:
```bash
ML_ENV_PYTHON="$HOME/MyDevelopment/Environments/Me_Labeller_venv/ml_env/bin/python3"
```

### Problem: "VoiceSpeechHealth import failed"
**Check**:
```bash
~/MyDevelopment/Environments/Me_Labeller_venv/ml_env/bin/python3 -c "import voicespeechhealth; print('OK')"
```

If this fails, reinstall VoiceSpeechHealth in ml_env:
```bash
source ~/MyDevelopment/Environments/Me_Labeller_venv/ml_env/bin/activate
pip install voicespeechhealth
```

### Problem: "No WAV files found"
**Solution**: Check your input directory path. The script only processes `*.wav` files (lowercase extension).

### Problem: Script produces no output
**Check the log file**:
```bash
cat parselmouth_extraction.log
```

### Problem: Conda still interfering
**Nuclear option** - Completely deactivate conda before running:
```bash
# In your terminal
conda deactivate  # Run multiple times until you see no (base)
./run_parselmouth_extraction.sh
```

---

## Performance Expectations

### Typical Processing Times:
- **0.1-second windows**: ~10 windows per second
- **90-second audio file**: ~15-20 seconds processing time
- **5-minute audio file**: ~1-2 minutes processing time

### Memory Usage:
- Small files (<5 min): ~200-300 MB
- Large files (>10 min): ~500 MB - 1 GB

---

## Integration with Your Workflow

### For Batch Processing Multiple Files:
```bash
# Process an entire directory
./run_parselmouth_extraction.sh ~/Audio/concatenated_wavs ~/Audio/parselmouth_features

# Process multiple directories
for dir in ~/Audio/Batch*; do
    ./run_parselmouth_extraction.sh "$dir" "${dir}_features"
done
```

### For CSA Analysis Integration:
According to your notes, you previously used:
1. **WavConcatenator** - to create per-speaker concatenated files
2. **VoiceSpeechHealth_deploy2.py** - to extract features
3. **me-md-id script** - to label which speaker is which

This new script fits into step 2, but extracts features in 0.1-second windows instead of whole-file aggregates.

---

## Why This Solution Works

1. **Explicit Python path**: Bypasses `which python3` entirely
2. **Unsets conda variables**: Prevents conda from hijacking the environment
3. **Direct execution**: Uses `/path/to/python3 script.py` instead of relying on shebang
4. **Validation**: Checks everything before starting
5. **Logging**: Both console and file output for debugging

---

## Next Steps After Successful Run

1. **Verify output**: Check that CSV files are created and populated
2. **Validate features**: Ensure feature values are in reasonable ranges
3. **Scale up**: Process your full dataset
4. **Integrate results**: Use with me-md-id script to label speakers

---

## Questions or Issues?

If problems persist after trying these solutions:

1. Run `./diagnose_environment.sh` and share the output
2. Check `parselmouth_extraction.log` for detailed errors
3. Verify your ml_env installation:
   ```bash
   ~/MyDevelopment/Environments/Me_Labeller_venv/ml_env/bin/python3 -m pip list | grep -i voice
   ```

---

## Files You Need

Copy these three files to your working directory:
1. `diagnose_environment.sh` - Diagnostic tool
2. `run_parselmouth_extraction.sh` - Launcher script
3. `parselmouth_standalone_improved.py` - Feature extractor

All scripts are designed to work together and handle the conda interference issue automatically.
