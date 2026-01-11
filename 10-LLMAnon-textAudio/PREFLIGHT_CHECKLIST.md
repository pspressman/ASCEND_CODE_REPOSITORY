# Pre-Flight Checklist - LLMAnon Round 2

Before running the redaction script, verify all these items:

## ✅ Prerequisites

### 1. Ollama Running
- [ ] Ollama service is running
- Test: Open command prompt and run: `ollama list`
- Start if needed: `ollama serve`

### 2. Model Installed
- [ ] llama3.2 (or your preferred model) is installed
- Test: `ollama list` should show llama3.2
- Install if needed: `ollama pull llama3.2`

### 3. Virtual Environment
- [ ] Venv exists at: `C:\LocalData\LLMAnon\llmanon_env`
- [ ] Can activate with: `C:\LocalData\LLMAnon\llmanon_env\Scripts\activate`

### 4. Required Files Exist

Input files:
- [ ] `C:\LocalData\LLMAnon\LLM-Anon-reviewed.csv`
- [ ] `C:\LocalData\ASCEND_PHI\DeID\CliniDeID_organized copy\` (directory with transcripts)
- [ ] `C:\LocalData\LLMAnon\Batch1_CSA_Timestamps\text_not_found_Batch1.txt`
- [ ] `C:\LocalData\LLMAnon\Batch2_Focused_Timestamps\text_not_found_Batch2.txt`

Script files:
- [ ] `C:\LocalData\LLMAnon\llmanon_round2_redactor.py`
- [ ] `C:\LocalData\LLMAnon\run_llmanon_round2.bat` (or .ps1)

### 5. Output Directory
- [ ] Output directory ready: `C:\LocalData\LLMAnon\Round2_Transcripts`
- (Will be created automatically if it doesn't exist)

## 🚀 Ready to Run?

### Option 1: Batch File (Recommended)
```
Double-click: C:\LocalData\LLMAnon\run_llmanon_round2.bat
```

### Option 2: PowerShell
```
Right-click run_llmanon_round2.ps1 → Run with PowerShell
```

### Option 3: Manual Command
```
cd C:\LocalData\LLMAnon
llmanon_env\Scripts\activate
python llmanon_round2_redactor.py --csv "C:\LocalData\LLMAnon\LLM-Anon-reviewed.csv" --transcripts "C:\LocalData\ASCEND_PHI\DeID\CliniDeID_organized copy" --text_not_found_batch1 "C:\LocalData\LLMAnon\Batch1_CSA_Timestamps\text_not_found_Batch1.txt" --text_not_found_batch2 "C:\LocalData\LLMAnon\Batch2_Focused_Timestamps\text_not_found_Batch2.txt" --output "C:\LocalData\LLMAnon\Round2_Transcripts" --model llama3.2
```

## 📊 What to Watch For

### During Processing
- ✅ "Connected to Ollama: llama3.2"
- ✅ "Total text-not-found entries: ~2990"
- ✅ "Validated PHI to process: ~60"
- ✅ "Found X labeled transcripts" and "Found X resynthesis transcripts"

### Expected Progress
- Processing should take ~5-10 minutes
- You'll see progress like: [1/60], [2/60], etc.
- Each item shows: participant ID, PHI text preview, and success status

### Red Flags
- ❌ "Ollama not accessible" → Start Ollama first
- ❌ "Model not found" → Install model: `ollama pull llama3.2`
- ❌ "CSV not found" → Check CSV path
- ❌ "Transcript directory not found" → Check transcripts path

## ✓ Success Indicators

After completion, you should see:
- ✅ "ROUND 2 REDACTION COMPLETE"
- ✅ ~48 transcript pairs found
- ✅ ~60 labeled redactions
- ✅ ~55-60 resynth redactions
- ✅ Output folder created: `C:\LocalData\LLMAnon\Round2_Transcripts`
- ✅ Summary JSON: `Round2_Processing_Summary.json`

## 🔍 Quick Validation

After processing, spot-check a few files:

1. Open a `*_LLMAnon_labeled.txt` file
   - Look for `[LLMAnon-Redacted]` markers
   - Verify original PHI text is gone
   - Confirm `[*** NAME ***]` tags are preserved

2. Open a `*_LLMAnon_resynth.txt` file
   - Look for Ollama-generated replacements
   - Should read naturally (not obvious redactions)
   - Confirm `[*** NAME ***]` tags are preserved

3. Check the summary JSON
   - Review statistics
   - Verify expected numbers match reality

## 🆘 Troubleshooting

### Issue: "Text not in transcript"
- **Expected**: Some items won't be in transcripts (JSON/transcript version mismatch)
- **Normal range**: 5-10 items
- **Action**: Check summary to see if count is reasonable

### Issue: "Ollama failures"
- **Expected**: A few failures are normal
- **Normal range**: <5 failures
- **Fallback used**: `[LLMAnon-Redacted-Fallback]`
- **Action**: If >10 failures, check Ollama is responsive

### Issue: "Transcript pairs not found"
- **Check**: Participant IDs in CSV match file naming in transcript directory
- **Use**: Substring matching should find them
- **Action**: Review a few missing IDs manually to verify naming pattern

---

**Ready?** Check all boxes above, then run the script!
