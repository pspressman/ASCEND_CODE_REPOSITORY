# Quick Start Guide - Testing the Comprehensive Pipeline

## Before Running on Full Dataset

**CRITICAL:** Test on a small subset first!

---

## Step 1: Create Test Folder

```bash
# Create a test folder
mkdir -p /Volumes/Databackup2025/TEST_DEID_SUBSET

# Copy 10-20 files (mix of audio and analysis files)
cp /Volumes/Databackup2025/ClinWavFiles_Anon_unparsed/some_audio.wav /Volumes/Databackup2025/TEST_DEID_SUBSET/
cp /Volumes/Databackup2025/ClinWavFiles_Anon_unparsed/analysis.csv /Volumes/Databackup2025/TEST_DEID_SUBSET/
# ... repeat for a few more files
```

**What to include in test:**
- 5-10 audio files with `MRN-date` filenames
- 2-3 CSV files that reference those audio files
- 1 Excel file if you have any
- 1 text/log file if you have any

---

## Step 2: Modify Script for Testing

Edit `comprehensive_deid_pipeline.py`:

```python
# Change line 19 to point to TEST folder:
source_directories = [
    "/Volumes/Databackup2025/TEST_DEID_SUBSET",
]

# Change line 23 to separate test output:
output_base_dir = "/Volumes/Databackup2025/TEST_DEID_OUTPUT"
```

---

## Step 3: Run Test

```bash
# Navigate to where you saved the script
cd ~/Desktop  # or wherever you put it

# Run it
python3 comprehensive_deid_pipeline.py
```

**Expected test output:**
```
======================================================================
COMPREHENSIVE PHI DE-IDENTIFICATION PIPELINE
======================================================================

[1/4] Loading key file...
✓ Loaded 150 patients from key file
[2/4] Initializing PHI detector...
[3/4] Initializing content de-identifier...
[4/4] Starting comprehensive de-identification...

======================================================================
Processing: /Volumes/Databackup2025/TEST_DEID_SUBSET
======================================================================

======================================================================
DE-IDENTIFICATION COMPLETE
======================================================================
Files with renamed filenames:    8
Files with content processed:    5
Files copied unchanged:          2
Total PHI replacements:          47
Errors:                          0

Time elapsed:                    0:00:23
```

---

## Step 4: Verify Output

### A. Check filenames were renamed:
```bash
ls /Volumes/Databackup2025/TEST_DEID_OUTPUT/TEST_DEID_SUBSET/
```

**Should see:**
- `P001-03-01-22.wav` (not `1234-01-15-22.wav`)
- `P078-05-17-22.wav` (not `5678-03-10-22.wav`)

### B. Check CSV content was replaced:

```bash
# Original CSV
head /Volumes/Databackup2025/TEST_DEID_SUBSET/analysis.csv

# De-identified CSV
head /Volumes/Databackup2025/TEST_DEID_OUTPUT/TEST_DEID_SUBSET/analysis.csv
```

**Original might show:**
```csv
participant,file,score
1234,1234-01-15-22.wav,85
```

**De-identified should show:**
```csv
participant,file,score
P001,P001-03-01-22.wav,85
```

### C. Check audit log:

```bash
cat /Users/peterpressman/MyDevelopment/Logs/PHI_Replacements_Audit.csv
```

**Should show every replacement:**
```csv
timestamp,file,location,original_phi,replacement
2024-10-31T10:23:45,analysis.csv,[participant][0],1234,P001
2024-10-31T10:23:45,analysis.csv,[file][0],1234-01-15-22.wav,P001-03-01-22.wav
```

### D. Verify originals are untouched:

```bash
# Original test folder should be EXACTLY the same
ls /Volumes/Databackup2025/TEST_DEID_SUBSET/
```

---

## Step 5: Manual Spot Check

1. **Open a de-identified CSV in Excel/Numbers**
   - Look for any MRNs (4-digit numbers matching your patients)
   - Should all be UIDs now

2. **Open a de-identified text file**
   - Search for any known MRNs
   - Should all be UIDs

3. **Check dates are shifted**
   - Pick a known patient (e.g., MRN 1234)
   - Find their date shift in key file (e.g., +45 days)
   - Verify dates in output are shifted by that amount

---

## Step 6: Validate No False Positives

Check if any non-PHI numbers were incorrectly replaced:

**Example false positives to watch for:**
- Phone numbers: `555-1234` shouldn't become `555-P001`
- Scores/measurements: `Score: 1234` → Should this be `Score: P001`?
- File sizes: `1234 KB` shouldn't change

**How to check:**
- Review the audit log CSV
- Look for suspicious replacements
- If found, adjust the pattern detection in the script

---

## Step 7: If Test Looks Good

1. ✅ Filenames correctly renamed
2. ✅ CSV/Excel content replaced
3. ✅ Dates properly shifted
4. ✅ No false positives
5. ✅ Originals untouched
6. ✅ Audit log comprehensive

**Then proceed to full dataset:**

```python
# Edit comprehensive_deid_pipeline.py back to:
source_directories = [
    "/Volumes/Databackup2025/ClinWavFiles_Anon_unparsed",
]

output_base_dir = "/Volumes/Databackup2025/DeidentifiedData"
```

**Run on full dataset:**
```bash
python3 comprehensive_deid_pipeline.py
```

---

## Troubleshooting Test

### "No UID found for MRN XXXX"
- That MRN isn't in your key file
- Check if it needs zero-padding
- Add to key file or exclude from test

### "File not found"
- Check paths are correct
- Verify test folder exists
- Check drive is mounted

### Very slow
- Normal for first run (Python compiling)
- Large Excel files are slow
- Should be faster on subsequent runs

### Errors in log
- Check the debug log: `FileRenamingDebugLog.txt`
- Look for specific error messages
- Most errors won't stop the whole process

---

## After Successful Test

**Delete test output:**
```bash
rm -rf /Volumes/Databackup2025/TEST_DEID_OUTPUT
```

**Run on full dataset with confidence!**

---

## Test Checklist

Before running on full dataset, verify:

- [ ] Test completed without errors
- [ ] Filenames renamed correctly
- [ ] CSV content replaced correctly
- [ ] Excel content replaced correctly (if tested)
- [ ] No false positives in replacements
- [ ] Dates shifted by correct amount
- [ ] Originals completely untouched
- [ ] Audit log looks comprehensive
- [ ] Folder structure preserved in output
- [ ] No PHI remaining in any output files

**If all checked:** Ready for full dataset! 🚀

**If any unchecked:** Review that item, adjust script if needed, re-test.

---

## Expected Full Dataset Processing

Assuming ~5,000 files:
- **Time:** 1-2 hours
- **Disk space:** Need 2x source size
- **Can interrupt:** Ctrl+C is safe (originals never touched)
- **Can resume:** Delete output and restart
- **System usage:** Low CPU, moderate disk I/O

**You can work while it runs!**

---

## Questions During Test?

Common issues:

**Q: Should `1234` in a score be replaced?**
A: Depends on context. If it's actually a score, no. If it's participant ID, yes. Check the audit log to see what was replaced and decide if pattern needs adjustment.

**Q: What if dates don't look right?**
A: Verify the date shift in your key file is correct for that patient.

**Q: Can I stop the test midway?**
A: Yes, Ctrl+C is safe. Originals never touched.

**Q: Test found no PHI to replace?**
A: Either:
1. Test files don't have PHI (try different files)
2. Pattern not matching (check filename format)
3. MRNs not in key file (verify key file)
