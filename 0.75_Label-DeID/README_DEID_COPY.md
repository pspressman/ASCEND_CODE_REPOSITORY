# Safe De-identification Script - COPY VERSION

## Critical Changes from Original

### Original Script (DANGEROUS):
```python
os.rename(original_path, new_path)  # DESTROYS ORIGINALS!
```

### New Script (SAFE):
```python
shutil.copy2(source_path, output_path)  # PRESERVES ORIGINALS!
```

---

## What This Script Does

1. **Reads your key file** (`deid_key.csv`) containing:
   - MRN → UID mappings
   - Patient-specific date shifts

2. **Scans source folders** for files matching pattern: `MRN-MM-DD-YY.extension`

3. **For each file**:
   - Extracts MRN and date from filename
   - Looks up patient's UID and date shift
   - Calculates shifted date
   - **COPIES** file to output directory with new name: `UID-shifted_date.extension`

4. **Maintains complete folder structure** in output directory

5. **NEVER touches originals** - they remain 100% intact

---

## Configuration (Lines 16-26)

```python
# Input: Where your original files live (NOT MODIFIED)
folders_to_search = [
    "/Volumes/Databackup2025/ClinWavFiles_Anon_unparsed",
    # Add more folders as needed
]

# Output: Where de-identified copies go
output_base_dir = "/Volumes/Databackup2025/DeidentifiedData"

# Key file with MRN→UID mappings and date shifts
key_file = "/Users/peterpressman/Desktop/deid_key.csv"
```

---

## Key Features

### ✅ Safety
- **Originals untouched**: All source files remain exactly as they were
- **Folder structure preserved**: Output mirrors input directory tree
- **Comprehensive logging**: Everything documented in log files
- **Duplicate handling**: Auto-adds counter if filename collision occurs

### 📊 Tracking
- **Progress indicators**: Shows files processed every 10 copies
- **Detailed logs**: 
  - `/Users/peterpressman/MyDevelopment/Logs/FileDeIDCopyLog.txt` (detailed)
  - `/Users/peterpressman/MyDevelopment/Logs/DeIDCopySummary.txt` (summary)
  - `/Users/peterpressman/MyDevelopment/Logs/UnmatchedFilesLog.csv` (unmatched)

### 🔍 Validation
- **Space check**: Warns if <10 GB free before starting
- **File matching**: Logs any files that don't match expected pattern
- **Error handling**: Continues processing even if individual files fail

---

## Usage

```bash
# Make executable (first time only)
chmod +x deid_copy_safe.py

# Run the script
python3 deid_copy_safe.py
```

---

## Output Structure Example

**Original:**
```
/Volumes/Databackup2025/ClinWavFiles_Anon_unparsed/
├── SubfolderA/
│   ├── 1234-01-15-22.wav
│   └── 1234-02-20-22.wav
└── SubfolderB/
    └── 5678-03-10-22.wav
```

**Output:**
```
/Volumes/Databackup2025/DeidentifiedData/ClinWavFiles_Anon_unparsed/
├── SubfolderA/
│   ├── P001-03-01-22.wav  (shifted +45 days, MRN→UID)
│   └── P001-04-06-22.wav  (shifted +45 days, MRN→UID)
└── SubfolderB/
    └── P078-05-17-22.wav  (shifted +68 days, MRN→UID)
```

---

## Expected Performance

- **Speed**: ~50-200 files/minute (depends on file sizes)
- **Space**: Requires duplicate storage (plan for 2x original size)
- **Safety**: 100% - originals never modified

---

## Before You Run

1. ✅ **Verify key file exists**: `/Users/peterpressman/Desktop/deid_key.csv`
2. ✅ **Check input folders exist**: Verify all paths in `folders_to_search`
3. ✅ **Confirm output space**: Ensure sufficient space on `/Volumes/Databackup2025`
4. ✅ **Create log directory**: `mkdir -p /Users/peterpressman/MyDevelopment/Logs`

---

## After Running

1. **Check the summary log** for statistics
2. **Review unmatched files log** if any files weren't processed
3. **Spot-check** a few output files to verify de-identification
4. **Verify originals** remain unchanged
5. **Once confident**, originals can be archived/deleted (but keep them safe initially!)

---

## Troubleshooting

### "Key file could not be loaded"
- Check path to `deid_key.csv`
- Verify CSV has columns: `mrn`, `UID`, `date_shift_days`

### "Folder not found"
- Verify paths in `folders_to_search`
- Check drive `/Volumes/Databackup2025` is mounted

### "Permission denied"
- May need to run with `sudo` or check folder permissions
- Verify write access to output directory

### "Insufficient space"
- Clear space on target drive
- Or change `output_base_dir` to different location

---

## Next Steps: Moving to Nexus

Once de-identification is complete on Databackup2025:

1. **Verify outputs** are correct
2. **Order additional drives** for Nexus/Synology
3. **Transfer de-identified data** to Nexus
4. **Delete** de-identified copies from Databackup2025 (keep originals!)
5. **Securely archive or destroy** original identified data per protocol

---

## Important Notes

⚠️ **This script only handles FILENAMES, not audio content**
- For audio content de-identification, use `phi_inplace_deidentifier_MASTER_CSV.py`
- This script is for organizing and protecting file metadata

🔐 **Security**
- Keep `deid_key.csv` secured and backed up separately
- This is the ONLY way to re-identify data if needed

📝 **Compliance**
- Document this de-identification process
- Maintain audit trails
- Follow your IRB/ethics protocols
