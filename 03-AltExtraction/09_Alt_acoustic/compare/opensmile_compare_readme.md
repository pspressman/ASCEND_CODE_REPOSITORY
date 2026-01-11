# openSMILE ComParE Feature Extractor

**Code Folder:** 04_TrackA_openSMILE_LOCUS (extended)

**Script:** `opensmile_compare_extractor.py`

**Computer:** LOCUS

**Status:** ⚠️ TO DO - Run on all cohorts after eGeMAPS extraction complete

---

## Purpose

Extract openSMILE ComParE_2016 comprehensive acoustic features from audio files that already have Pyannote diarization. This script complements the existing eGeMAPS extraction by adding ~6,000 additional ComParE features while dropping the 88 eGeMAPS features already extracted.

**Single extraction level:**
- **Full-task aggregated** - Per-speaker + non-diarized combined (ALL tasks)

**No per-segment or granular extraction** - ComParE is aggregated functionals only.

---

## Inputs

**Required Pre-existing Files:**
- Audio files: `.wav` format, task-separated
- Diarization: `{basename}.rttm` (from Step 03_Foundation_Pyannote_LOCUS)

**Input Structure:**
```
/path/to/input/
├── participant_001/
│   ├── participant_001_spontspeech.wav
│   ├── participant_001_spontspeech.rttm
│   ├── participant_001_gfp.wav
│   ├── participant_001_gfp.rttm
│   ├── participant_001_mse.wav
│   ├── participant_001_mse.rttm
│   └── ...
```

**Cohorts to Process:**
- CSA Clinical Main (n=250)
- CSA Research Cross-sectional (n=50)
- CSA Research Longitudinal
- 004 (n=20)
- LIIA (n=100)
- clinic_plus (n=150)

*All cohorts that already have eGeMAPS features*

---

## Outputs

**Per audio file, creates:**

### Full-Task Aggregated (ALL tasks)
- `{basename}_combined_compare_features.csv` - Non-diarized (all speakers combined)
- `{basename}_speaker{N}_compare_features.csv` - Per-speaker diarized

**Output Structure:**
```
/path/to/output/
├── participant_001/
│   ├── participant_001_spontspeech_combined_compare_features.csv
│   ├── participant_001_spontspeech_speaker0_compare_features.csv
│   ├── participant_001_spontspeech_speaker1_compare_features.csv
│   ├── participant_001_gfp_combined_compare_features.csv
│   ├── participant_001_gfp_speaker0_compare_features.csv
│   ├── participant_001_mse_combined_compare_features.csv
│   ├── participant_001_mse_speaker0_compare_features.csv
│   └── ...
```

**Next Step Destinations:**
- Non-diarized features → `CSAforProton/04_AcousticFeatures/FromRawAudio/openSMILE_ComParE/{cohort}/Complete/NonDiarized/`
- Per-speaker features → `CSAforProton/04_AcousticFeatures/FromRawAudio/openSMILE_ComParE/{cohort}/Complete/PerSpeaker/`

---

## Features Extracted (~6,000 features)

### ComParE_2016 Feature Set
The Computational Paralinguistics Challenge (ComParE) 2016 feature set is the most comprehensive openSMILE feature set, containing:

**130 Low-Level Descriptors (LLDs) including:**
- Energy and spectral features
- Voicing related features (F0, jitter, shimmer)
- MFCCs 1-14
- Spectral harmonics, roll-off, flux
- Formants 1-3 with amplitudes
- Line spectral pairs (LSPs)
- And many more...

**49 Functionals applied to each LLD:**
- Statistical measures: mean, std, min, max, range
- Percentiles: 1%, 99%, range
- Quartiles and inter-quartile ranges
- Regression coefficients (linear and quadratic)
- Moments: skewness, kurtosis
- Slopes, offsets
- And more...

**Total:** 130 LLDs × 49 functionals = ~6,370 features

### eGeMAPS Columns Dropped
The script automatically drops the 88 eGeMAPS features already extracted in the initial run to avoid redundancy:
- F0 (pitch) features
- Formant frequencies and bandwidths (F1, F2, F3)
- Loudness features
- Spectral features (alpha ratio, Hammarberg index, slopes)
- Jitter and shimmer
- HNR (Harmonic-to-Noise Ratio)
- MFCCs 1-4

**Result:** ~6,000 unique ComParE features not in eGeMAPS

---

## Dependencies

**Python Version:**
- Python 3.8+

**Python Libraries:**
- `opensmile` - Feature extraction
- `soundfile` - Audio I/O
- `torchaudio` - Audio processing (optional, for consistency)
- `numpy`, `pandas` - Data manipulation

**Installation:**
```bash
pip install opensmile soundfile torchaudio numpy pandas
```

**openSMILE Binary:**
- Should be automatically managed by Python openSMILE package
- No separate installation needed (unlike old openSMILE)

---

## Usage

**Basic usage:**
```bash
python opensmile_compare_extractor.py \
  --input /Users/peterpressman/Desktop/CompleteClinicAudioOutput \
  --output /Users/peterpressman/Desktop/CompleteClinicAudioOutput
```

**Process specific cohort:**
```bash
python opensmile_compare_extractor.py \
  --input /Volumes/Databackup2025/Data/CSA_Research \
  --output /Users/peterpressman/Desktop/CSA_Research_ComParE
```

---

## Features

### Resume Capability
The script automatically detects existing outputs and skips them:
- ✓ File exists and has content → Skip
- ✗ File missing or empty → Extract features

### Audio Preprocessing
- Converts stereo to mono automatically
- Skips segments shorter than 0.1 seconds
- Concatenates all speaker segments before feature extraction

### eGeMAPS Column Removal
- Automatically identifies and removes 88 eGeMAPS features
- Prevents feature redundancy with existing eGeMAPS extraction
- Leaves ~6,000 unique ComParE features

---

## Processing Time

**Per file estimates:**

**Non-diarized combined:**
- 2-minute audio: ~30-60 seconds
- 5-minute audio: ~1-2 minutes

**Per-speaker diarized:**
- 1 speaker, 2 minutes: ~30-60 seconds
- 2 speakers, 5 minutes: ~1-2 minutes per speaker

**Total per file (2 speakers, 5 minutes):** ~3-5 minutes

**Bottleneck:** ComParE feature set is computationally expensive due to large number of features.

**Comparison to eGeMAPS:**
- eGeMAPS (88 features): ~20-40 seconds per file
- ComParE (~6,000 features): ~3-5 minutes per file
- ~5-7x slower than eGeMAPS

---

## Output File Descriptions

### 1. `{basename}_combined_compare_features.csv`
Non-diarized ComParE features extracted from entire audio file (all speakers combined).

**Format:** Single row with ~6,000 feature columns

**Used for:** Comparison of diarized vs. non-diarized feature extraction impact on ML performance

### 2. `{basename}_speaker{N}_compare_features.csv`
Per-speaker aggregated ComParE features from all segments for that speaker.

**Format:** Single row with ~6,000 feature columns

**Used for:** Primary ML training/testing with comprehensive acoustic feature set

---

## Comparison with Other Feature Sets

### ComParE vs. eGeMAPS
- **eGeMAPS:** 88 carefully selected minimal acoustic parameter set
  - Interpretable, standardized
  - Good for cross-study comparisons
  - Already extracted
- **ComParE:** ~6,000 comprehensive feature set
  - Includes eGeMAPS features + much more
  - Better for ML when feature selection is applied
  - Captures more subtle acoustic phenomena

### ComParE vs. Parselmouth
- **Overlap:** Some similar features (F0, formants, HNR)
- **ComParE unique:** More spectral features, LSPs, extensive functionals
- **Parselmouth unique:** CPP, specific voice quality measures, speaking rate
- **Purpose:** Different algorithmic implementations for methodological comparison

### ComParE vs. Librosa
- **Overlap:** Spectral features, MFCCs, energy
- **ComParE unique:** Voice quality features, more extensive functionals
- **Librosa unique:** Chroma features, different MFCC implementation
- **Purpose:** Methodological comparison across toolkits

---

## Methodological Rationale

### Why Extract ComParE?
1. **Comprehensive coverage:** Most extensive standard acoustic feature set
2. **Redundancy for robustness:** Multiple related features increase reliability
3. **Feature selection:** Let ML algorithms choose most informative features
4. **Benchmark standard:** ComParE is widely used in paralinguistics research
5. **Methodological comparison:** Compare against eGeMAPS minimal set

### Why Drop eGeMAPS Columns?
1. **Avoid redundancy:** Already have these features from previous extraction
2. **Storage efficiency:** ~6,000 new features vs. ~6,370 total
3. **Clear separation:** Can analyze eGeMAPS vs. ComParE-unique performance
4. **Processing time:** Slightly faster by not re-extracting known features

### Non-Diarized vs. Diarized Comparison
Critical for ASCEND methodological questions:
- Does speaker separation improve ML performance?
- Is the computational cost of diarization justified?
- Do participant-only features outperform combined features?
- How much does investigator speech contaminate features?

---

## Status by Cohort

| Cohort | eGeMAPS Status | ComParE Status | Notes |
|--------|---------------|----------------|-------|
| CSA Clinical Main | ✅ DONE | 🔲 TO DO | ~1000 files, ~50-80 hours processing |
| CSA Research X-sect | ✅ DONE | 🔲 TO DO | ~250 files, ~12-20 hours processing |
| CSA Research Long | ⚠️ IN PROGRESS | 🔲 TO DO | Wait for eGeMAPS completion |
| 004 | ⚠️ NEEDS VERIFICATION | 🔲 TO DO | Verify eGeMAPS first |
| LIIA | ✅ DONE | 🔲 TO DO | ~300 files, ~15-25 hours processing |
| clinic_plus | ⚠️ PARTIAL | 🔲 TO DO | Complete remaining eGeMAPS first |

**Estimated Total Processing Time:** ~100-150 hours across all cohorts

**Strategy:** Run overnight/weekend batches on LOCUS

---

## Troubleshooting

**"Missing diarization" warnings:**
- Ensure RTTM files exist in same directory as audio files
- Check that Step 03 (Pyannote diarization) completed successfully

**Memory issues:**
- ComParE is memory-intensive due to large feature set
- Close other applications
- Process in smaller batches if needed
- LOCUS has sufficient RAM for this task

**Slow processing:**
- Expected - ComParE extracts ~6,000 features
- Use resume capability to process in chunks
- Run overnight for large cohorts

**OpenSMILE errors:**
- Check opensmile package installation: `pip install --upgrade opensmile`
- Verify audio files are valid WAV format
- Check logs for specific error messages

---

## Integration with Existing Pipeline

### Relationship to eGeMAPS Extraction
- **eGeMAPS:** Already extracted via `optimized_audio_processor.py`
- **ComParE:** Additional features extracted by this script
- **No overlap:** eGeMAPS columns explicitly dropped from ComParE output
- **Compatible:** Both can be used together or separately for ML

### File Organization
```
CSAforProton/04_AcousticFeatures/FromRawAudio/
├── openSMILE_eGeMEPS/          # From optimized_audio_processor.py
│   ├── CSA-Clinical/Complete/
│   ├── CSA-Research/Complete/
│   └── ...
├── openSMILE_ComParE/          # From this script
│   ├── CSA-Clinical/Complete/
│   │   ├── NonDiarized/
│   │   └── PerSpeaker/
│   ├── CSA-Research/Complete/
│   └── ...
```

---

## Notes

- **Run AFTER eGeMAPS extraction** - Requires existing RTTM files
- **Computationally expensive** - Plan for long processing times
- **Resume-friendly** - Can interrupt and restart without losing progress
- **Comprehensive feature set** - Best for ML with feature selection
- **Methodological comparison** - Compare against eGeMAPS, Parselmouth, Librosa
- **Non-diarized important** - Critical for assessing diarization impact

---

## Checklist

- [ ] Script tested on sample files
- [ ] opensmile package installed and working
- [ ] eGeMAPS extraction complete for target cohort
- [ ] Input paths verified (existing RTTM + audio files)
- [ ] Output directory structure created
- [ ] CSA Clinical Main - ComParE extraction
- [ ] CSA Research X-sect - ComParE extraction
- [ ] CSA Research Long - ComParE extraction
- [ ] 004 - ComParE extraction
- [ ] LIIA - ComParE extraction
- [ ] clinic_plus - ComParE extraction
- [ ] Verify eGeMAPS columns dropped successfully
- [ ] Verify outputs in CSAforProton structure
- [ ] Compare performance: eGeMAPS vs. ComParE vs. other toolkits
- [ ] Ready for ML training
