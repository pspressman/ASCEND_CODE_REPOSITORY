# Comprehensive De-identification Pipeline

## The Problem You Identified

After weeks of analysis, PHI isn't just in **filenames** - it's propagated throughout your workflow:

1. **CSV files** with columns containing:
   - `1234-01-15-22.wav` (file references)
   - `1234` (just MRN)
   - `Patient: 1234 visited on 01-15-22`

2. **Excel spreadsheets** with:
   - Analysis results referencing MRNs
   - Participant IDs in headers/data
   - File paths with PHI

3. **JSON files** with:
   - Metadata containing MRNs
   - Configuration files with patient identifiers

4. **Text/log files** with:
   - Processing logs mentioning MRNs
   - Notes referencing patients

---

## What This Pipeline Does

This is a **comprehensive 2-stage de-identification system**:

### Stage 1: Filename De-identification
- Renames files: `MRN-MM-DD-YY.ext` → `UID-shifted_date.ext`
- Maintains folder structure
- Logs all filename changes

### Stage 2: Content De-identification
- **Scans inside files** for PHI patterns:
  - `MRN-date` patterns → `UID-shifted_date`
  - `MRN only` patterns → `UID`
- Processes multiple file types:
  - **CSV files**: All cells scanned and replaced
  - **Excel files**: All sheets, all cells scanned
  - **JSON files**: All string values scanned
  - **Text files**: Entire content scanned

### Critical Features
- ✅ **Never modifies originals** - everything is copied
- ✅ **Maintains folder structure** exactly
- ✅ **Comprehensive audit log** - every replacement tracked
- ✅ **Smart pattern detection** - avoids false positives
- ✅ **Handles multiple formats** - flexible date/MRN patterns

---

## PHI Detection Patterns

### Pattern 1: MRN-Date (e.g., `1234-01-15-22`)
**Detected in:**
- Filenames
- CSV cell values
- Excel cells
- JSON string values
- Text file content

**Replaced with:** `UID-shifted_date` (e.g., `P001-03-01-22`)

### Pattern 2: MRN Only (e.g., `1234`)
**Detected in:**
- Any 3-4 digit number that matches a valid MRN
- Uses word boundaries to avoid false positives
- Only if not part of an MRN-date pattern

**Replaced with:** `UID` (e.g., `P001`)

### Smart Detection
- Won't replace `1234` in phone numbers like `555-1234-5678`
- Won't replace if already part of `1234-01-15-22` pattern
- Only replaces MRNs that exist in your key file

---

## Configuration

### Lines 16-30: What You Need to Set

```python
# Your key file (MRN → UID mappings + date shifts)
key_file = "/Users/peterpressman/Desktop/deid_key.csv"

# Directories to process (originals - NOT modified)
source_directories = [
    "/Volumes/Databackup2025/ClinWavFiles_Anon_unparsed",
    # Add more as needed
]

# Output directory for de-identified copies
output_base_dir = "/Volumes/Databackup2025/DeidentifiedData"

# File types to scan for PHI content (in addition to renaming)
content_scan_extensions = [
    '.csv', '.xlsx', '.xls', '.txt', '.json', '.md', '.log'
]
```

---

## File Processing Examples

### Example 1: CSV with PHI in cells

**Original CSV** (`analysis_results.csv`):
```csv
participant,file,score,date
1234,1234-01-15-22.wav,85,01/15/22
5678,5678-03-10-22.wav,92,03/10/22
```

**De-identified CSV**:
```csv
participant,file,score,date
P001,P001-03-01-22.wav,85,03/01/22
P078,P078-05-17-22.wav,92,05/17/22
```

### Example 2: JSON with nested PHI

**Original JSON** (`metadata.json`):
```json
{
  "study_data": {
    "participant_1234": {
      "audio_file": "1234-01-15-22.wav",
      "mrn": "1234",
      "date": "01/15/22"
    }
  }
}
```

**De-identified JSON**:
```json
{
  "study_data": {
    "participant_P001": {
      "audio_file": "P001-03-01-22.wav",
      "mrn": "P001",
      "date": "03/01/22"
    }
  }
}
```

### Example 3: Text log file

**Original** (`processing.log`):
```
Processing participant 1234
File: 1234-01-15-22.wav
MRN 1234 completed successfully
```

**De-identified**:
```
Processing participant P001
File: P001-03-01-22.wav
MRN P001 completed successfully
```

---

## Output Structure

```
/Volumes/Databackup2025/DeidentifiedData/
├── ClinWavFiles_Anon_unparsed/
│   ├── SubfolderA/
│   │   ├── P001-03-01-22.wav          (renamed file)
│   │   ├── analysis_results.csv        (content replaced)
│   │   └── metadata.json               (content replaced)
│   └── SubfolderB/
│       ├── P078-05-17-22.wav          (renamed file)
│       └── processing_log.txt          (content replaced)
```

---

## Audit Trail

### Replacement Log (`PHI_Replacements_Audit.csv`)
Every single replacement is logged:

```csv
timestamp,file,location,original_phi,replacement
2024-10-31T10:23:45,analysis.csv,[participant][0],1234,P001
2024-10-31T10:23:45,analysis.csv,[file][0],1234-01-15-22.wav,P001-03-01-22.wav
2024-10-31T10:23:46,metadata.json,:study_data.mrn,1234,P001
```

**This allows you to:**
- Verify all replacements
- Audit de-identification process
- Track exactly what changed where
- Demonstrate compliance

---

## Safety Features

### 1. Copy-Only Operations
```python
# NEVER does this to originals:
os.rename(original_path, new_path)  # ❌

# ALWAYS does this:
shutil.copy2(source_path, output_path)  # ✅
```

### 2. Pattern Validation
- Only replaces MRNs that exist in key file
- Uses word boundaries to avoid partial matches
- Won't replace numbers in middle of larger numbers

### 3. Collision Handling
If `P001-03-01-22.wav` already exists:
- Creates `P001-03-01-22_001.wav`
- Increments counter until unique

### 4. Comprehensive Logging
- Debug log: Every operation
- Replacement log: Every PHI change
- Summary log: Overall statistics

---

## Usage

```bash
# Install required packages (if needed)
pip install pandas openpyxl numpy --break-system-packages

# Make executable
chmod +x comprehensive_deid_pipeline.py

# Run the pipeline
python3 comprehensive_deid_pipeline.py
```

**Expected output:**
```
======================================================================
COMPREHENSIVE PHI DE-IDENTIFICATION PIPELINE
Filenames + Content (CSVs, Excel, JSON, Text)
======================================================================

[1/4] Loading key file...
✓ Loaded 150 patients from key file
[2/4] Initializing PHI detector...
[3/4] Initializing content de-identifier...
[4/4] Starting comprehensive de-identification...

======================================================================
Processing: /Volumes/Databackup2025/ClinWavFiles_Anon_unparsed
======================================================================

  Progress: 500 files processed...

======================================================================
DE-IDENTIFICATION COMPLETE
======================================================================
Files with renamed filenames:    45
Files with content processed:    127
Files copied unchanged:          328
Total PHI replacements:          892
Errors:                          0

Time elapsed:                    0:05:23

Originals preserved in:          /Volumes/Databackup2025/ClinWavFiles_Anon_unparsed
De-identified copies:            /Volumes/Databackup2025/DeidentifiedData
Replacement audit log:           /Users/peterpressman/MyDevelopment/Logs/PHI_Replacements_Audit.csv
======================================================================
```

---

## Performance Expectations

### Speed
- **Renaming only**: ~200-500 files/minute
- **Content scanning**: ~50-150 files/minute (depends on file size)
- **CSV/Excel files**: Slower (need to parse every cell)
- **Large files**: May take several minutes each

### Resources
- **Memory**: Minimal for most files
- **Large Excel files**: May use several hundred MB temporarily
- **Disk space**: Need 2x original data size

### Estimated Times
- 1,000 files (mostly audio): ~5-10 minutes
- 1,000 files (50% CSVs/Excel): ~15-25 minutes
- 5,000 files mixed: ~1-2 hours

---

## Before You Run - Checklist

- [ ] **Key file exists** and has columns: `mrn`, `UID`, `date_shift_days`
- [ ] **Source directories exist** and are accessible
- [ ] **Sufficient disk space** (at least 2x source size + 10GB buffer)
- [ ] **Log directory created**: `mkdir -p /Users/peterpressman/MyDevelopment/Logs`
- [ ] **Backup key file** somewhere secure (this is the ONLY way to re-identify!)
- [ ] **Test on small subset** first (create test folder with 10-20 files)

---

## Validation Steps

After running, you should:

1. **Check summary log** for statistics
2. **Open replacement audit log** - verify replacements make sense
3. **Spot-check files**:
   - Open a de-identified CSV - verify MRNs are UIDs
   - Check filenames are renamed correctly
   - Verify dates are shifted consistently
4. **Verify originals** are untouched (important!)
5. **Test a few files** with your analysis scripts

---

## Troubleshooting

### "No UID found for MRN XXXX"
- MRN in files but not in key file
- Check if MRN needs zero-padding (e.g., `234` → `0234`)
- Add missing MRNs to key file

### "Permission denied"
- Check read permissions on source files
- Check write permissions on output directory
- May need `sudo` for system directories

### False Positives (numbers incorrectly replaced)
- Adjust pattern matching in `PHIDetector` class
- Add exclusion patterns for known false positives
- Use more specific regex patterns

### Very slow processing
- Large Excel files can be slow
- Consider processing in smaller batches
- Check if antivirus is scanning files

---

## What This DOESN'T Handle

### Audio Content
This pipeline only handles **metadata and filenames**, not audio content itself.

For audio content de-identification, you still need:
- `phi_inplace_deidentifier_MASTER_CSV.py` (your other script)

### Images/PDFs with embedded text
- Would need OCR to detect PHI in images
- PDFs would need text extraction

### Encrypted files
- Can't scan inside encrypted containers

---

## Next Steps

1. **Test on small subset** (10-20 files)
2. **Verify output quality**
3. **Run on full dataset**
4. **Review audit logs**
5. **Transfer to Nexus** (after SSD upgrade arrives)
6. **Securely archive/destroy** originals per protocol

---

## Key Advantages Over Original Script

| Feature | Original | New Pipeline |
|---------|----------|--------------|
| Handles filenames | ✅ | ✅ |
| Handles CSV content | ❌ | ✅ |
| Handles Excel content | ❌ | ✅ |
| Handles JSON content | ❌ | ✅ |
| Handles text files | ❌ | ✅ |
| Audit logging | Basic | Comprehensive |
| Pattern detection | Filename only | Everywhere |
| False positive prevention | None | Smart boundaries |
| Preserves originals | ❌ Renames | ✅ Copies |

---

## Questions?

Common concerns addressed:

**Q: Will this slow down my system?**
A: Not significantly - it's I/O bound, not CPU bound. You can work while it runs.

**Q: What if I need to stop it midway?**
A: Safe to Ctrl+C - originals never touched. Just restart and it continues.

**Q: Can I run it multiple times?**
A: Yes, but will create duplicates. Delete output directory first or use different output path.

**Q: How do I verify it worked correctly?**
A: Check the audit log CSV - every replacement is documented with before/after values.

**Q: What if I find PHI it missed?**
A: Add patterns to `PHIDetector` class or report the pattern for future enhancement.
