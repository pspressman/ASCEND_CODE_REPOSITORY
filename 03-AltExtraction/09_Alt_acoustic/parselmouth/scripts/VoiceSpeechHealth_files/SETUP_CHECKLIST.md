# Parselmouth Extraction Setup - Checklist

Use this checklist to ensure successful setup and execution.

---

## □ PHASE 1: File Preparation

### Download & Organize Files

- [ ] Download all 6 files from outputs directory:
  - [ ] `setup_parselmouth.sh`
  - [ ] `diagnose_environment.sh`
  - [ ] `run_parselmouth_extraction.sh`
  - [ ] `parselmouth_standalone_improved.py`
  - [ ] `README_PARSELMOUTH_FIX.md`
  - [ ] `SOLUTION_SUMMARY.md`

- [ ] Move files to working directory:
  ```bash
  # Suggested location
  cd ~/Desktop/AudioThreeTest/parseltest
  # Or wherever you keep your scripts
  ```

- [ ] Verify all files present:
  ```bash
  ls -la *.sh *.py *.md
  ```

---

## □ PHASE 2: Environment Verification

### Test ml_env Installation

- [ ] Verify ml_env Python exists:
  ```bash
  ls -la ~/MyDevelopment/Environments/Me_Labeller_venv/ml_env/bin/python3
  ```

- [ ] Test VoiceSpeechHealth import:
  ```bash
  ~/MyDevelopment/Environments/Me_Labeller_venv/ml_env/bin/python3 \
    -c "import voicespeechhealth; print('✅ VoiceSpeechHealth OK')"
  ```

- [ ] If import fails, reinstall:
  ```bash
  source ~/MyDevelopment/Environments/Me_Labeller_venv/ml_env/bin/activate
  pip install voicespeechhealth
  deactivate
  ```

### Check Test File

- [ ] Verify test WAV exists:
  ```bash
  ls -lh ~/Desktop/AudioThreeTest/parseltest/CSA_MAC_Audacity_2-003-8_20SEP2022_Paralang.wav
  ```

- [ ] If missing, create test directory or adjust paths in scripts

---

## □ PHASE 3: Initial Setup

### Run Setup Script

- [ ] Make setup script executable:
  ```bash
  chmod +x setup_parselmouth.sh
  ```

- [ ] Run setup:
  ```bash
  ./setup_parselmouth.sh
  ```

- [ ] Verify setup script completed:
  - [ ] All scripts now executable (✓✓✓ in output)
  - [ ] Diagnostics ran successfully
  - [ ] Output directory created

### Review Diagnostic Results

From `setup_parselmouth.sh` output, verify:

- [ ] VIRTUAL_ENV shows ml_env path (or [not set] - OK either way)
- [ ] CONDA_DEFAULT_ENV shows base (warning) or [not set] (good)
- [ ] ml_env Python exists: ✅
- [ ] VoiceSpeechHealth imports successfully: ✅
- [ ] Required packages all show: ✅
  - soundfile
  - pandas
  - parselmouth
- [ ] Test file exists: ✅

**If any checks fail, STOP and troubleshoot before proceeding**

---

## □ PHASE 4: Test Run

### Single File Test

- [ ] Run extraction on test file:
  ```bash
  ./run_parselmouth_extraction.sh
  ```

### Monitor Progress

Watch for these success indicators:

- [ ] "✅ Environment Check" appears
- [ ] "VoiceSpeechHealth imported successfully" in stderr
- [ ] "Processing: CSA_MAC_Audacity..." appears
- [ ] "Audio loaded: X samples" appears
- [ ] Progress updates appear: "Processed 100/933 windows..."
- [ ] "✓ Saved XXX windows (25 features)" appears
- [ ] "✅ Extraction completed successfully!" appears

### Verify Output

- [ ] Check output file created:
  ```bash
  ls -lh ~/Desktop/AudioThreeTest/parseltest/output/*.csv
  ```

- [ ] Verify CSV has data:
  ```bash
  wc -l ~/Desktop/AudioThreeTest/parseltest/output/*.csv
  # Should show hundreds of rows (933 for test file + 1 header)
  ```

- [ ] Check CSV structure:
  ```bash
  head -n 2 ~/Desktop/AudioThreeTest/parseltest/output/*.csv
  ```
  Should show:
  - First line: Column headers (29 columns total)
  - Second line: First window data

- [ ] Verify log file created:
  ```bash
  cat parselmouth_extraction.log
  ```

---

## □ PHASE 5: Validation

### Feature Value Sanity Check

- [ ] Open CSV in spreadsheet or Python
- [ ] Spot-check feature values are reasonable:
  - [ ] mean_F0: typically 80-250 Hz
  - [ ] mean_dB: typically 40-80 dB
  - [ ] HNR_dB: typically 5-25 dB
  - [ ] Speaking_Rate: typically 2-6 syllables/sec
  - [ ] Formants (F1, F2): typically 200-3000 Hz

- [ ] Check for excessive NaN/missing values:
  ```bash
  # Count non-empty rows
  wc -l output/*.csv
  # Should be close to expected window count
  ```

### Performance Check

From log output, verify:

- [ ] Processing time reasonable (~15-30 sec for 93-sec file)
- [ ] Memory usage acceptable (check Activity Monitor if concerned)
- [ ] No error messages in log

---

## □ PHASE 6: Scale Up (After Successful Test)

### Process Full Dataset

- [ ] Identify your full input directory:
  ```
  Example: ~/Audio/concatenated_wavs/
  ```

- [ ] Identify your desired output directory:
  ```
  Example: ~/Audio/parselmouth_features/
  ```

- [ ] Run on full dataset:
  ```bash
  ./run_parselmouth_extraction.sh \
    ~/Audio/concatenated_wavs \
    ~/Audio/parselmouth_features
  ```

### Monitor Batch Processing

- [ ] Check progress periodically:
  ```bash
  # Count completed files
  ls ~/Audio/parselmouth_features/*.csv | wc -l
  
  # Compare to input files
  ls ~/Audio/concatenated_wavs/*.wav | wc -l
  ```

- [ ] Review log for errors:
  ```bash
  grep "ERROR" parselmouth_extraction.log
  grep "failed" parselmouth_extraction.log
  ```

---

## □ PHASE 7: Integration

### Integrate with Existing Workflow

- [ ] Identify previous results from VoiceSpeechHealth_deploy2.py
- [ ] Compare file counts and feature names
- [ ] Note differences:
  - Old: 1 row per file (aggregate)
  - New: ~600 rows per minute (0.1-sec windows)

### Label Speakers (me-md-id script)

- [ ] Run me-md-id script to label which speaker is which
- [ ] Add speaker labels to feature CSVs
- [ ] Verify speaker assignments correct

### Prepare for Analysis

- [ ] Organize features by:
  - [ ] Participant ID
  - [ ] Task type
  - [ ] Speaker role (if applicable)
  - [ ] Diagnosis group

- [ ] Merge with metadata:
  - [ ] Demographics
  - [ ] Neuropsych scores
  - [ ] Diagnosis

---

## □ PHASE 8: Documentation

### Record Your Setup

- [ ] Note any path customizations made
- [ ] Document which directories you used
- [ ] Save example output files
- [ ] Record processing times for your dataset

### Update Project Notes

- [ ] Add to VoiceSpeechHealth_process_notes
- [ ] Update project timeline
- [ ] Note completion of Batch 1 / Batch 2

---

## □ TROUBLESHOOTING REFERENCE

### If something goes wrong:

1. **No output at all**
   - [ ] Run: `./diagnose_environment.sh`
   - [ ] Check: `parselmouth_extraction.log`
   - [ ] Verify: ml_env Python path in `run_parselmouth_extraction.sh`

2. **Import errors**
   - [ ] Test: VoiceSpeechHealth import manually
   - [ ] Reinstall: VoiceSpeechHealth in ml_env
   - [ ] Check: Python version (should be 3.x)

3. **Conda interference**
   - [ ] Deactivate: Run `conda deactivate` multiple times
   - [ ] Verify: `echo $CONDA_DEFAULT_ENV` shows nothing
   - [ ] Use: Launcher script (handles this automatically)

4. **Path errors**
   - [ ] Verify: Input directory exists and has .wav files
   - [ ] Check: Output directory permissions
   - [ ] Adjust: Paths in script if ml_env is elsewhere

5. **Feature extraction fails**
   - [ ] Check: WAV file format (should be standard PCM WAV)
   - [ ] Verify: File not corrupted (can play in media player)
   - [ ] Review: Specific error in log file

---

## □ SUCCESS CRITERIA

Your setup is successful when:

- [x] ✅ All diagnostic checks pass
- [x] ✅ Test file processes without errors
- [x] ✅ CSV output contains expected number of windows
- [x] ✅ Feature values are in reasonable ranges
- [x] ✅ Log file shows detailed processing information
- [x] ✅ Can run on multiple files/directories
- [x] ✅ Processing time acceptable for your dataset size

---

## □ NEXT STEPS (Post-Setup)

After successful extraction:

- [ ] Process remaining CSA data (Batch 2)
- [ ] Compare Parselmouth vs OpenSMILE features
- [ ] Train ML models on both feature sets
- [ ] Analyze temporal dynamics (unique to windowed approach)
- [ ] Correlate with neuropsych assessments
- [ ] Write up results by April 30, 2025

---

## QUICK REFERENCE COMMANDS

```bash
# Setup (once)
chmod +x setup_parselmouth.sh
./setup_parselmouth.sh

# Test run
./run_parselmouth_extraction.sh

# Full dataset
./run_parselmouth_extraction.sh /input/dir /output/dir

# Check diagnostics
./diagnose_environment.sh

# View log
cat parselmouth_extraction.log

# Count completed files
ls output/*.csv | wc -l
```

---

## HELP & DOCUMENTATION

- **Full usage guide**: `README_PARSELMOUTH_FIX.md`
- **Solution overview**: `SOLUTION_SUMMARY.md`
- **Visual workflow**: `WORKFLOW_DIAGRAM.txt`
- **This checklist**: `SETUP_CHECKLIST.md`

---

**Last Updated**: October 21, 2025  
**Purpose**: Parselmouth feature extraction with environment fix  
**Status**: Production ready
