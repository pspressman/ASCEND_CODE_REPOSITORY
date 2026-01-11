# LLMAnon Bridge Scripts - CRITICAL FIX

## The Problem (0 JSON files found out of 1959)

**Root Cause:** Filename matching logic was using **exact dictionary key lookup** after normalization, but the normalization was inconsistent and failed.

### Why It Failed:

```python
# FILE-basename in CSV: "5746613-1-18-18"
# Actual JSON filename: "5746613-1-18-18.deid.piiCategoryTag.json"

# OLD APPROACH (BROKEN):
normalized_id = normalize_filename("5746613-1-18-18")  # → "5746613-1-18-18"
json_file = json_index[normalized_id]  # KeyError! Not in index

# WHY? The index was built with normalized filenames:
# "5746613-1-18-18.deid.piiCategoryTag.json" → normalize → "5746613-1-18-18"
# But there were issues with multi-extension normalization order
```

## The Solution: Substring Matching

**Key Insight from Past Conversations:** 
- Don't rely on normalization magic - filenames vary wildly
- The FILE-basename is CONTAINED in the actual filename
- Build indexes with ACTUAL filenames, then do substring search

### What Changed:

1. **Build indexes with ACTUAL filenames (no normalization)**
   ```python
   # OLD: json_index[normalized_id] = path
   # NEW: json_index[actual_filename] = path
   
   # Example:
   json_index["5746613-1-18-18.deid.piiCategoryTag.json"] = Path(...)
   ```

2. **Do substring matching when looking up**
   ```python
   # FILE-basename from CSV: "5746613-1-18-18"
   search_id = file_basename  # Use directly, no normalization!
   
   # Search for ANY filename that contains this ID
   for filename in json_index.keys():
       if search_id in filename:
           return json_index[filename]
   ```

3. **If multiple matches, prefer shortest (most specific)**

## Files Modified:

### 1. LLMAnon_bridge_Batch1_CSA_FIXED.py
**Changes:**
- `build_json_index()` - No normalization, use actual filenames as keys
- `build_wav_index()` - No normalization, use actual filenames as keys  
- Added `find_file_by_substring()` - Substring matching function
- Main loop - Use substring matching instead of exact key lookup

**Lines Changed:**
- Lines 146-183: JSON index builder (removed normalization)
- Lines 186-223: WAV index builder (removed normalization)
- Lines 77-108: Added substring matching function
- Lines 592-616: Replaced exact lookup with substring search

### 2. LLMAnon_bridge_Batch2_Focused_FIXED.py
**Changes:** Same as Batch1, but with Batch 2 configuration:
- JSON directory: `focused_audio_anon\WordTimings_focused`
- Output directory: `Batch2_Focused_Timestamps`

## Why This Approach Works:

1. **No pattern dependency** - Works regardless of filename patterns
2. **No normalization required** - Avoids normalization bugs
3. **Handles all extensions** - `.json`, `.deid.piiCategoryTag.json`, etc.
4. **Exact match preferred** - Substring search naturally finds exact matches first
5. **Proven approach** - This is what worked in previous successful scripts

## Testing:

**Before:**
```
JSON files found:              0
JSON files missing:            3176
```

**Expected After:**
```
JSON files found:              1959  (or more)
JSON files missing:            <200  (legitimate missing files)
```

## Usage:

```bash
# Batch 1 (CSA WordStamps)
python LLMAnon_bridge_Batch1_CSA_FIXED.py

# Batch 2 (Focused WordTimings)
python LLMAnon_bridge_Batch2_Focused_FIXED.py
```

## Key Learnings:

1. **Trust the past conversations** - The solution was already documented
2. **Don't normalize indexes** - Use actual filenames
3. **Substring matching is more robust** than normalization magic
4. **Exact matches happen naturally** with substring search
5. **Pattern matching doesn't work** - filenames vary too much

## Related Past Issues:

From conversation history:
- "Pattern matching has been tried and wasted hours"
- "The FILE-basename will contain the kind of output as described, but the whole POINT is that base file names can vary wildly"
- "SO STOP TELLING ME ABOUT HOW FILE NAME PATTERNS DONT LINE UP IT DOESN'T MATTER"

The lesson: **Build indexes with actual filenames, search by substring**. Don't try to be clever with normalization.
