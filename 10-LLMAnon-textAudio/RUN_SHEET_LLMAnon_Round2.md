# LLMAnon Round 2 Anonymization - Complete Run Sheet

**Date:** October 31, 2025  
**Environment:** `deid_env` (already activated)  
**Total Runs:** 6 (2 bridge + 4 deidentifier)

---

## 📂 File Locations

### Scripts
- **Bridge (Batch 1):** `C:\LocalData\LLMAnon\LLMAnon_bridge_Batch1_CSA.py`
- **Bridge (Batch 2):** `C:\LocalData\LLMAnon\LLMAnon_bridge_Batch2_Focused.py`
- **Deidentifier:** `C:\LocalData\LLMAnon\phi_inplace_deidentifier_NESTED.py`

### Input Data
- **LLMAnon CSV:** `C:\LocalData\LLMAnon\LLM-Anon-reviewed.csv` (same for both batches)
- **JSON Batch 1:** `C:\LocalData\ASCEND_PHI\DeID\CSA_WordStamps` (nested)
- **JSON Batch 2:** `C:\LocalData\ASCEND_PHI\DeID\focused_audio_anon\WordTimings_focused` (nested)
- **Audio Clean:** `Z:\ASCEND_ANON\McAdams\CleanedSplicedAudio` (nested)
- **Audio Original:** `Z:\ASCEND_ANON\McAdams\OriginalSplicedAudio` (nested)

---

## 🎯 Execution Plan

### PHASE 1: Generate Timestamps (Bridge Scripts)

#### Run 1: Batch 1 Timestamps
```bash
cd C:\LocalData\LLMAnon
python LLMAnon_bridge_Batch1_CSA.py
```

**Output:**
```
C:\LocalData\LLMAnon\Batch1_CSA_Timestamps\
├── phi_timestamps_Batch1_CSA.csv           ← MAIN OUTPUT (feed to deidentifier)
├── bridge_Batch1_CSA_YYYYMMDD_HHMMSS.log
├── text_not_found_Batch1.txt               ← Expected (LLM false positives)
├── clinideid_markers_skipped_Batch1.txt    ← Already handled in Round 1
└── missing_files_Batch1.txt
```

**Expected:** Files in CSA_WordStamps will be found. Files in focused batch will show as "JSON not found" (normal).

---

#### Run 2: Batch 2 Timestamps
```bash
cd C:\LocalData\LLMAnon
python LLMAnon_bridge_Batch2_Focused.py
```

**Output:**
```
C:\LocalData\LLMAnon\Batch2_Focused_Timestamps\
├── phi_timestamps_Batch2_Focused.csv       ← MAIN OUTPUT (feed to deidentifier)
├── bridge_Batch2_Focused_YYYYMMDD_HHMMSS.log
├── text_not_found_Batch2.txt
├── clinideid_markers_skipped_Batch2.txt
└── missing_files_Batch2.txt
```

**Expected:** Files NOT found in Batch 1 will be found here.

---

### PHASE 2: Apply Audio Redactions (Deidentifier Runs)

⚠️ **CRITICAL:** The deidentifier now preserves nested folder structure!

#### Run 3: Batch 1 × Clean Audio
```bash
cd C:\LocalData\LLMAnon
python phi_inplace_deidentifier_NESTED.py \
  --audio_dir "Z:\ASCEND_ANON\McAdams\CleanedSplicedAudio" \
  --master_csv "C:\LocalData\LLMAnon\Batch1_CSA_Timestamps\phi_timestamps_Batch1_CSA.csv" \
  --output_dir "Z:\ASCEND_ANON\McAdams\LLMAnon_Round2\Batch1_CSA_Clean" \
  --method hybrid
```

**Output:**
```
Z:\ASCEND_ANON\McAdams\LLMAnon_Round2\Batch1_CSA_Clean\
├── [MIRRORED FOLDER STRUCTURE FROM INPUT]
│   ├── task1\
│   │   └── file1_HYBRID_deidentified.wav
│   └── task2\
│       └── file2_HYBRID_deidentified.wav
├── inplace_deidentification_summary.json
└── processing_metadata.csv
```

---

#### Run 4: Batch 1 × Original Audio
```bash
cd C:\LocalData\LLMAnon
python phi_inplace_deidentifier_NESTED.py \
  --audio_dir "Z:\ASCEND_ANON\McAdams\OriginalSplicedAudio" \
  --master_csv "C:\LocalData\LLMAnon\Batch1_CSA_Timestamps\phi_timestamps_Batch1_CSA.csv" \
  --output_dir "Z:\ASCEND_ANON\McAdams\LLMAnon_Round2\Batch1_CSA_Original" \
  --method hybrid
```

**Output:**
```
Z:\ASCEND_ANON\McAdams\LLMAnon_Round2\Batch1_CSA_Original\
├── [MIRRORED FOLDER STRUCTURE FROM INPUT]
└── (same structure as Run 3)
```

---

#### Run 5: Batch 2 × Clean Audio
```bash
cd C:\LocalData\LLMAnon
python phi_inplace_deidentifier_NESTED.py \
  --audio_dir "Z:\ASCEND_ANON\McAdams\CleanedSplicedAudio" \
  --master_csv "C:\LocalData\LLMAnon\Batch2_Focused_Timestamps\phi_timestamps_Batch2_Focused.csv" \
  --output_dir "Z:\ASCEND_ANON\McAdams\LLMAnon_Round2\Batch2_Focused_Clean" \
  --method hybrid
```

**Output:**
```
Z:\ASCEND_ANON\McAdams\LLMAnon_Round2\Batch2_Focused_Clean\
├── [MIRRORED FOLDER STRUCTURE FROM INPUT]
└── (nested structure preserved)
```

---

#### Run 6: Batch 2 × Original Audio
```bash
cd C:\LocalData\LLMAnon
python phi_inplace_deidentifier_NESTED.py \
  --audio_dir "Z:\ASCEND_ANON\McAdams\OriginalSplicedAudio" \
  --master_csv "C:\LocalData\LLMAnon\Batch2_Focused_Timestamps\phi_timestamps_Batch2_Focused.csv" \
  --output_dir "Z:\ASCEND_ANON\McAdams\LLMAnon_Round2\Batch2_Focused_Original" \
  --method hybrid
```

**Output:**
```
Z:\ASCEND_ANON\McAdams\LLMAnon_Round2\Batch2_Focused_Original\
├── [MIRRORED FOLDER STRUCTURE FROM INPUT]
└── (nested structure preserved)
```

---

## 📊 Final Output Structure

```
Z:\ASCEND_ANON\McAdams\LLMAnon_Round2\
├── Batch1_CSA_Clean\
│   └── [NESTED FOLDERS MIRRORING CleanedSplicedAudio]
│       └── *_HYBRID_deidentified.wav
├── Batch1_CSA_Original\
│   └── [NESTED FOLDERS MIRRORING OriginalSplicedAudio]
│       └── *_HYBRID_deidentified.wav
├── Batch2_Focused_Clean\
│   └── [NESTED FOLDERS MIRRORING CleanedSplicedAudio]
│       └── *_HYBRID_deidentified.wav
└── Batch2_Focused_Original\
    └── [NESTED FOLDERS MIRRORING OriginalSplicedAudio]
        └── *_HYBRID_deidentified.wav
```

**CRITICAL:** Folder structure from input is preserved in output!

---

## ✅ Quality Checks After Each Run

### After Bridge Runs:
```bash
# Check CSV was created
ls -lh C:\LocalData\LLMAnon\Batch1_CSA_Timestamps\phi_timestamps_Batch1_CSA.csv

# Check how many timestamps generated
wc -l C:\LocalData\LLMAnon\Batch1_CSA_Timestamps\phi_timestamps_Batch1_CSA.csv

# Check for errors
grep "ERROR" C:\LocalData\LLMAnon\Batch1_CSA_Timestamps\bridge_Batch1_CSA_*.log
```

### After Deidentifier Runs:
```bash
# Verify nested structure preserved
tree /F "Z:\ASCEND_ANON\McAdams\LLMAnon_Round2\Batch1_CSA_Clean" | more

# Count output files
dir /S /B "Z:\ASCEND_ANON\McAdams\LLMAnon_Round2\Batch1_CSA_Clean\*.wav" | find /C ".wav"

# Check summary
type "Z:\ASCEND_ANON\McAdams\LLMAnon_Round2\Batch1_CSA_Clean\inplace_deidentification_summary.json"
```

---

## ⏱️ Expected Runtime

**Per Bridge Run:** 10-15 minutes  
**Per Deidentifier Run:** 1-3 hours (depending on dataset size)  
**Total:** ~6-14 hours for all 6 runs

---

## 🚨 Troubleshooting

### Bridge Script Issues

**"JSON not found"**
- Expected if file is in the other batch
- Check `missing_files_Batch1.txt` - should appear in Batch 2 logs

**"Text not found in JSON"**
- EXPECTED! LLM overzealousness
- Check `text_not_found_Batch1.txt` - review but don't panic

**"CliniDeID markers skipped"**
- GOOD! Means `[***NAME***]` patterns detected correctly
- Check `clinideid_markers_skipped_Batch1.txt` for list

### Deidentifier Issues

**"Audio file not found"**
- Verify WAV exists in specified audio_dir
- Check that filename in CSV matches exactly
- Remember: recursive search is active

**"Output folder structure wrong"**
- Make sure you're using `phi_inplace_deidentifier_NESTED.py` (not the old version)
- Check that input paths are correct

---

## 📝 Notes

1. **Run bridge scripts FIRST** - you need timestamps before deidentifying
2. **Bridge runs are independent** - can't combine because JSON locations differ
3. **Deidentifier preserves nesting** - folder structure from input → output
4. **Same audio, different JSONs** - Batch 1 and 2 share audio files but have different JSON timestamp sources
5. **All scripts use recursive search** - `os.walk()` finds files in nested directories

---

## 🎯 Success Criteria

✅ Bridge Batch 1: Generates CSV with timestamps for CSA files  
✅ Bridge Batch 2: Generates CSV with timestamps for Focused files  
✅ Deidentifier runs complete without errors  
✅ Output preserves nested folder structure  
✅ Listen to sample outputs - confirm additional redactions applied  
✅ Total files processed = files in both JSON directories  

---

**Ready to execute! Start with Run 1 (Bridge Batch 1).**
