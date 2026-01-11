# LLMAnon Round 2 Pipeline - Complete Package

## 🎯 What Was Fixed

### CRITICAL FIX: Nested Folder Structure Preservation

**The Problem:**
Original deidentifier script flattened output:
```python
# OLD CODE (WRONG):
output_path = output_dir / f"{participant_id}_HYBRID_deidentified.wav"
# Result: All files dumped in one directory → CHAOS
```

**The Fix:**
Modified deidentifier preserves folder structure:
```python
# NEW CODE (CORRECT):
relative_path = audio_path.relative_to(audio_dir)
output_path = output_dir / relative_path.parent / output_filename
output_path.parent.mkdir(parents=True, exist_ok=True)
# Result: Nested structure mirrored from input → ORGANIZED
```

**Example:**
```
INPUT:  Z:\...\CleanedSplicedAudio\GrandfatherPassage\file.wav
OUTPUT: Z:\...\LLMAnon_Round2\Batch1_CSA_Clean\GrandfatherPassage\file_HYBRID_deidentified.wav
        └── FOLDER STRUCTURE PRESERVED! ✅
```

---

## 📦 Complete File Manifest

### 1. **phi_inplace_deidentifier_NESTED.py** (18KB)
**Purpose:** Modified deidentifier that preserves nested folder structure  
**Key Change:** Lines 327-345 - calculates relative paths and recreates directory tree  
**Use:** For all 4 deidentifier runs  
**Location:** Save to `C:\LocalData\LLMAnon\phi_inplace_deidentifier_NESTED.py`

---

### 2. **LLMAnon_bridge_Batch1_CSA.py** (26KB)
**Purpose:** Generate timestamps for Batch 1 (CSA files)  
**Configured For:**
- JSON: `C:\LocalData\ASCEND_PHI\DeID\CSA_WordStamps`
- Output: `C:\LocalData\LLMAnon\Batch1_CSA_Timestamps\`
**Features:**
- ✅ Mixed JSON format support (top-level `words` + `segments→words`)
- ✅ CliniDeID marker auto-detection
- ✅ Multi-word phrase matching
- ✅ Recursive search (os.walk)
**Location:** Save to `C:\LocalData\LLMAnon\LLMAnon_bridge_Batch1_CSA.py`

---

### 3. **LLMAnon_bridge_Batch2_Focused.py** (26KB)
**Purpose:** Generate timestamps for Batch 2 (Focused files - the overlooked batch)  
**Configured For:**
- JSON: `C:\LocalData\ASCEND_PHI\DeID\focused_audio_anon\WordTimings_focused`
- Output: `C:\LocalData\LLMAnon\Batch2_Focused_Timestamps\`
**Features:** Same as Batch 1, just different JSON directory  
**Location:** Save to `C:\LocalData\LLMAnon\LLMAnon_bridge_Batch2_Focused.py`

---

### 4. **RUN_SHEET_LLMAnon_Round2.md** (7.8KB)
**Purpose:** Step-by-step execution guide for all 6 runs  
**Contents:**
- Complete command-line instructions
- Expected outputs for each run
- Quality check commands
- Troubleshooting guide
- Runtime estimates
**Location:** Save to `C:\LocalData\LLMAnon\RUN_SHEET_LLMAnon_Round2.md`

---

### 5. **README_LLMAnon_Bridge.md** (13KB)
**Purpose:** General documentation for the bridge script  
**Contents:**
- How the script works
- Configuration guide
- CliniDeID marker detection explanation
- Multi-word phrase matching details
- Testing strategy
**Location:** Reference document

---

### 6. **WHY_THIS_SCRIPT_WINS.md** (4.2KB)
**Purpose:** Comparison with old IT_to_PHI script  
**Contents:**
- Feature comparison table
- Why mixed JSON format support is critical
- Why CliniDeID detection is essential
**Location:** Reference document

---

### 7. **LLMAnon_to_timestamps_bridge.py** (26KB)
**Purpose:** Generic/unconfigured version (for reference)  
**Note:** Use the batch-specific versions instead  
**Location:** Reference only

---

## 🚀 Quick Start Guide

### Step 1: Download All Files
Save to `C:\LocalData\LLMAnon\`:
- `phi_inplace_deidentifier_NESTED.py`
- `LLMAnon_bridge_Batch1_CSA.py`
- `LLMAnon_bridge_Batch2_Focused.py`
- `RUN_SHEET_LLMAnon_Round2.md`

### Step 2: Verify Environment
```bash
# Activate environment
conda activate deid_env

# Verify you're in the right place
cd C:\LocalData\LLMAnon
```

### Step 3: Execute Pipeline
Follow `RUN_SHEET_LLMAnon_Round2.md` step by step:
1. Run bridge for Batch 1
2. Run bridge for Batch 2
3. Run deidentifier 4 times (2 batches × 2 audio types)

---

## 🎯 The Complete Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ INPUT                                                         │
├─────────────────────────────────────────────────────────────┤
│ • LLMAnon CSV (all reviewed)                                 │
│ • JSON Batch 1 (CSA) + JSON Batch 2 (Focused)               │
│ • Audio: Clean + Original (shared by both batches)          │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: BRIDGE SCRIPTS (Generate Timestamps)               │
├─────────────────────────────────────────────────────────────┤
│ Run 1: LLMAnon_bridge_Batch1_CSA.py                         │
│        → phi_timestamps_Batch1_CSA.csv                       │
│                                                               │
│ Run 2: LLMAnon_bridge_Batch2_Focused.py                     │
│        → phi_timestamps_Batch2_Focused.csv                   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: DEIDENTIFIER (Apply Audio Redactions)              │
├─────────────────────────────────────────────────────────────┤
│ Run 3: Batch1 × Clean    → Batch1_CSA_Clean\               │
│ Run 4: Batch1 × Original → Batch1_CSA_Original\             │
│ Run 5: Batch2 × Clean    → Batch2_Focused_Clean\           │
│ Run 6: Batch2 × Original → Batch2_Focused_Original\         │
│                                                               │
│ ✅ All preserve nested folder structure!                    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ OUTPUT                                                        │
├─────────────────────────────────────────────────────────────┤
│ Z:\ASCEND_ANON\McAdams\LLMAnon_Round2\                      │
│ ├── Batch1_CSA_Clean\         (nested folders preserved)    │
│ ├── Batch1_CSA_Original\      (nested folders preserved)    │
│ ├── Batch2_Focused_Clean\     (nested folders preserved)    │
│ └── Batch2_Focused_Original\  (nested folders preserved)    │
└─────────────────────────────────────────────────────────────┘
```

---

## ✅ What Makes This Pipeline Bulletproof

### 1. **Nested Structure Preservation** ⭐⭐⭐
- Folder hierarchy from input → output maintained
- No chaos from flattened files
- Easy to verify completeness

### 2. **Mixed JSON Format Support** ⭐⭐⭐
- Handles BOTH WhisperX structures automatically
- Top-level `words` array (newer)
- `segments` → `words` (legacy)
- **This took 8 hours to debug - now it's built in!**

### 3. **CliniDeID Marker Detection** ⭐⭐
- Auto-skips `[***NAME***]` patterns
- Logs separately (not as errors)
- Clean diagnostic output

### 4. **Recursive Search Everywhere** ⭐⭐
- `os.walk()` used in all file discovery
- Finds files in deeply nested directories
- No manual subdirectory specification needed

### 5. **Multi-Word Phrase Matching** ⭐
- "won national award" matches as single unit
- Punctuation stripped automatically
- Returns ALL occurrences

### 6. **Comprehensive Logging** ⭐
- 4 separate log files per bridge run
- Distinguishes expected vs. actual errors
- Easy troubleshooting

---

## 🔍 Pre-Flight Checklist

Before running anything:
- [ ] All 3 Python scripts in `C:\LocalData\LLMAnon\`
- [ ] `deid_env` activated
- [ ] LLMAnon CSV exists: `C:\LocalData\LLMAnon\LLM-Anon-reviewed.csv`
- [ ] JSON directories accessible (both batches)
- [ ] Audio directories accessible (Clean + Original)
- [ ] Output directory writable: `Z:\ASCEND_ANON\McAdams\LLMAnon_Round2\`

---

## 📞 Support

If issues arise:
1. Check logs in timestamp output directories
2. Review `RUN_SHEET_LLMAnon_Round2.md` troubleshooting section
3. Verify recursive search is finding files (check log counts)
4. Compare expected vs actual file counts

---

**Everything is configured and ready. Just run the commands in the run sheet in order.**

**Total execution time: ~6-14 hours for all 6 runs.**

**Let's anonymize this data! 🚀**
