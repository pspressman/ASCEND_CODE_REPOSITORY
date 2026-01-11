# Optimized Audio Processor - Foundation Layer

**Code Folders:** 02_Foundation_WhisperX_LOCUS + 03_Foundation_Pyannote_LOCUS + 04_TrackA_openSMILE_LOCUS

**Script:** `optimized_audio_processor.py`

**Computer:** LOCUS

**Status:** ✅ Complete for most cohorts (CSA Clinical Main, CSA Research, LIIA, 004 - needs verification)  
**Remaining:** ~50 files in clinic_plus cohort

---

## Purpose

This script combines three foundational processing steps into one optimized pipeline:
1. **WhisperX Transcription** - Generates accurate transcripts with word-level timestamps
2. **Pyannote Diarization** - Identifies speaker segments (who spoke when)
3. **openSMILE eGeMEPS Extraction** - Extracts acoustic features per speaker

By combining these steps, models are loaded once and memory is efficiently managed between processing stages.

---

## Inputs

**Required:**
- Audio files: `.wav` format, task-separated (from Step 01_Foundation_TaskSplitting_Manual)
- Location: Can be anywhere, specified as first argument

**Expected Input Structure:**
```
/path/to/audio/root/
├── participant_001/
│   ├── participant_001_spontspeech.wav
│   ├── participant_001_gfp.wav
│   ├── participant_001_mse.wav
│   └── participant_001_picnic.wav
├── participant_002/
│   └── ...
```

**Current Input Locations by Cohort:**
- CSA Clinical Main: `/Volumes/Databackup2025/Data/CSAND/Clinic Recordings/`
- CSA Research: `/Volumes/Databackup2025/Data/CSA_Research/`
- 004: `/Volumes/Databackup2025/Data/004/`
- LIIA: `/Volumes/Databackup2025/Data/LIIA/`
- clinic_plus: `/Volumes/Databackup2025/Data/clinic_plus/` (~50 files remaining)

---

## Outputs

**Per audio file, creates:**
1. `{basename}_transcript_raw.json` - Raw WhisperX output with alignment
2. `{basename}.rttm` - Diarization in RTTM format (speaker timing)
3. `{basename}_transcript.txt` - Human-readable transcript with speakers
4. `{basename}_non_speech_segments.csv` - Pauses, gaps, overlaps
5. `{basename}_speaker{N}_features.csv` - openSMILE eGeMEPS features per speaker

**Output Structure:**
```
/path/to/output/root/
├── participant_001/
│   ├── participant_001_spontspeech_transcript_raw.json
│   ├── participant_001_spontspeech.rttm
│   ├── participant_001_spontspeech_transcript.txt
│   ├── participant_001_spontspeech_non_speech_segments.csv
│   ├── participant_001_spontspeech_speaker0_features.csv
│   ├── participant_001_spontspeech_speaker1_features.csv
│   ├── participant_001_gfp_transcript_raw.json
│   ├── participant_001_gfp.rttm
│   └── ...
```

**Current Output Location:**
- `/Users/peterpressman/Desktop/CompleteClinicAudioOutput/` (or specified as second argument)

**Next Step Inputs:**
- **Transcripts** → Multiple destinations:
  - `{basename}_transcript.txt` → Step 12 (NLP1 Coherence - CLINAMEN)
  - `{basename}_transcript.txt` → Step 13 (NLP2 Non-Coherence - NEXUS)
  - `{basename}_transcript_raw.json` → Step 20 (Transcript CliniDeID Label - NEXUS)
- **Diarization** → Step 06 (Diarization folder in CSAforProton - gold standard comparison)
- **Acoustic Features** → CSAforProton final destination:
  - `{basename}_speaker{N}_features.csv` → `04_AcousticFeatures/FromRawAudio/openSMILE_eGeMEPS/{cohort}/Complete/`

---

## Dependencies

**Python Version:**
- Python 3.8 (required)

**Python Libraries:**
- `whisperx` - ASR with alignment
- `torch` / `torchaudio` - Audio processing
- `pyannote.audio` - Speaker diarization
- `opensmile` - Acoustic feature extraction
- `pandas`, `numpy` - Data manipulation
- `tqdm` - Progress bars

**CRITICAL: Transformers Version Conflict**
- WhisperX requires `transformers==4.36.x` (approximately)
- FasterWhisper setups may install incompatible newer versions
- **If you get errors about transformers**, run:
  ```bash
  pip install transformers==4.36.2
  ```
- Do NOT upgrade transformers if this script is working
- Create separate virtual environments for WhisperX vs FasterWhisper if needed

**External Requirements:**
- OpenSMILE installed at `/Users/peterpressman/opensmile`
- HuggingFace token for Pyannote: `YOUR_HF_TOKEN_HERE`
- GPU/MPS support recommended (falls back to CPU if unavailable)

**Model Downloads (automatic on first run):**
- WhisperX large-v2
- Pyannote speaker-diarization-3.1

---

## Usage

**Basic usage:**
```bash
python optimized_audio_processor.py /path/to/audio/input /path/to/output
```

**Default paths (if no arguments provided):**
```bash
python optimized_audio_processor.py
# Uses:
# Input:  /Volumes/Databackup2025/Data/CSAND/Clinic Recordings/
# Output: /Users/peterpressman/Desktop/CompleteClinicAudioOutput
```

**Processing clinic_plus remaining files:**
```bash
python optimized_audio_processor.py \
  /Volumes/Databackup2025/Data/clinic_plus/ \
  /Users/peterpressman/Desktop/ClinicPlusAudioOutput
```

---

## Features

### Resume Capability
The script automatically detects partially processed files and resumes where it left off:
- ✓ Transcript exists → Skip transcription, load from JSON
- ✓ Diarization exists → Skip diarization, load from RTTM
- ✓ Speaker N features exist → Skip that speaker's feature extraction
- ✓ All outputs complete → Skip entire file

### Memory Management
- Models loaded once at initialization, reused for all files
- Garbage collection and cache clearing between files
- Automatic device detection (MPS → CUDA → CPU fallback)

### Turn-Taking Analysis
Automatically detects and classifies:
- **Speech segments** - Normal speaking turns
- **Pauses** - Within-speaker silence
- **Gaps** - Between-speaker silence
- **Overlaps** - Simultaneous speech (with controller identification)

### Progress Tracking
- Shows resume status for each file (what's done, what's needed)
- Real-time progress through diarization
- Per-file timing and overall statistics

---

## Output File Descriptions

### 1. `{basename}_transcript_raw.json`
WhisperX output with word-level timestamps and alignment. Contains:
- Detected language
- Segments with start/end times
- Word-level alignments
- Confidence scores

**Used by:** De-identification pipeline (Step 20)

### 2. `{basename}.rttm`
Speaker diarization in RTTM format:
```
SPEAKER filename 1 0.000 2.450 <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER filename 1 2.450 1.200 <NA> <NA> SPEAKER_01 <NA> <NA>
```

**Used by:** Manual comparison with gold standard TextGrids (subset)

### 3. `{basename}_transcript.txt`
Human-readable transcript with speakers:
```
Speaker 0: Tell me about your day.
Gap: 2.45 - 2.67
Speaker 1: Well, I woke up early this morning.
Pause: 5.12 - 5.34
Speaker 1: Then I had breakfast.
```

**Used by:** NLP1 (Step 12) and NLP2 (Step 13)

### 4. `{basename}_non_speech_segments.csv`
Turn-taking dynamics:
```csv
start,end,type,speaker,speakers,controller
2.45,2.67,gap,,,1
5.12,5.34,pause,1,,
7.89,8.01,overlap,,[0,1],1
```

**Used by:** Conversational analysis (future)

### 5. `{basename}_speaker{N}_features.csv`
openSMILE eGeMEPS features aggregated across all turns for that speaker:
- 88 acoustic features per speaker
- Aggregated statistics: mean, median, std, min, max, COV, IQR
- Features include: F0, jitter, shimmer, MFCCs, spectral features, energy

**Format:**
```csv
index,variable,value
F0semitoneFrom27.5Hz_sma3nz_amean,mean,4.532
F0semitoneFrom27.5Hz_sma3nz_amean,median,4.421
...
```

**Goes to:** `CSAforProton/04_AcousticFeatures/FromRawAudio/openSMILE_eGeMEPS/{cohort}/Complete/`

---

## Processing Time

**Per file (conversational, 2 speakers, ~5 minutes audio):**
- Transcription + Alignment: ~2-3 minutes
- Diarization: ~3-10 minutes (most expensive)
- Feature extraction: ~30-60 seconds
- **Total: ~6-14 minutes per file**

**Per file (non-conversational, 1 speaker, ~2 minutes audio):**
- Transcription + Alignment: ~1-2 minutes
- Diarization: ~2-5 minutes
- Feature extraction: ~20-30 seconds
- **Total: ~3-7 minutes per file**

**Resume saves time:**
- If transcript exists: Save ~2-3 minutes
- If diarization exists: Save ~3-10 minutes
- Can resume mid-file if interrupted

---

## Status by Cohort

| Cohort | Total Files | Status | Notes |
|--------|-------------|--------|-------|
| CSA Clinical Main | ~1000 (250 participants × 4 tasks) | ✅ COMPLETE | All outputs verified |
| CSA Research X-sect | ~250 (50 participants × 5 tasks) | ✅ COMPLETE | All outputs verified |
| CSA Research Long | ~100 | ⚠️ IN PROGRESS | Running tonight |
| 004 | ~80 (20 participants × 4 tasks) | ⚠️ NEEDS VERIFICATION | Sparse features found in forLocus/PrimeCSANDExtract/004 |
| LIIA | ~300 (100 participants × 3 tasks) | ✅ COMPLETE | All outputs verified |
| clinic_plus | ~600 (150 participants × 4 tasks) | ⚠️ PARTIAL | ~550 done, ~50 remaining |

---

## Next Steps for Outputs

### Immediate Next Steps (Parallel Processing):
1. **Transcripts** → Copy to multiple destinations:
   - CLINAMEN: `forCLINAMEN/` for NLP1 coherence processing
   - NEXUS: `forNEXUS/` for NLP2 processing + de-identification
   
2. **Acoustic Features** → Verify and move to CSAforProton:
   - From: `/Users/peterpressman/Desktop/CompleteClinicAudioOutput/`
   - To: `~/Desktop/CSAforProton/04_AcousticFeatures/FromRawAudio/openSMILE_eGeMEPS/{cohort}/Complete/`

3. **Diarization RTTM** → Subset to CSAforProton:
   - Manual gold standard subset → `~/Desktop/CSAforProton/06_Diarization/ManualTextGrids_GoldStandard/`
   - All Pyannote outputs → `~/Desktop/CSAforProton/06_Diarization/Pyannote/{cohort}/Complete/`

### Downstream Processing:
- Transcripts trigger **ALL** parallel tracks (A, B, C, D)
- Features go directly to final deliverable (numbers are de-identified)
- Audio files continue to Track B de-identification pipeline

---

## Troubleshooting

**"MPS not supported" warning:**
- Expected on some operations
- Automatically falls back to CPU
- No action needed

**Memory issues:**
- Reduce `batch_size` (default: 16)
- Process fewer files at once
- Restart script to clear memory between batches

**Sparse/incomplete features (like 004 cohort):**
- Check `forLocus/PrimeCSANDExtract/004/` for existing outputs
- Verify with `is_already_processed()` function
- May need to re-run on 004 cohort specifically

**Diarization takes forever:**
- Normal for long files or many speakers
- Progress bar shows status
- Cannot be easily optimized (model limitation)

---

## Notes

- **This script replaces 3 separate steps** - more efficient than running them separately
- **Memory optimized** - Models loaded once, aggressive garbage collection
- **Resume-friendly** - Can interrupt and restart without losing progress
- **Most cohorts complete** - Focus remaining time on clinic_plus and 004 verification
- **12-day deadline** - Features from this step are CRITICAL (numbers = de-identified = safe to transfer)

---

## Checklist

- [x] Code complete and tested
- [x] Resume capability working
- [x] Memory management optimized
- [x] CSA Clinical Main complete
- [x] CSA Research X-sect complete
- [x] LIIA complete
- [ ] CSA Research Longitudinal (running tonight)
- [ ] clinic_plus remaining ~50 files
- [ ] 004 verification/completion
- [ ] Move outputs to CSAforProton structure
- [ ] Ready for delivery to OHSU