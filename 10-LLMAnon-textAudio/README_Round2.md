# LLMAnon Round 2 Transcript Redaction

## What This Does

Takes the 60 validated LLMAnon PHI items (from your audio redaction work) and applies them to CliniDeID transcripts, creating TWO new versions:

1. **Labeled Version** (`*_LLMAnon_labeled.txt`): Simple `[LLMAnon-Redacted]` replacements
2. **Resynthesis Version** (`*_LLMAnon_resynth.txt`): Ollama-generated contextual replacements

## Quick Start

### Option 1: Quick Runner (Easiest)
```bash
python run_round2_redaction.py
```

This uses pre-configured paths. Just review and confirm.

### Option 2: Manual Command
```bash
python llmanon_round2_redactor.py \
    --csv "C:\LocalData\LLMAnon\LLM-Anon-reviewed.csv" \
    --transcripts "C:\LocalData\ASCEND_PHI\DeID\CliniDeID_organized copy" \
    --text_not_found_batch1 "C:\LocalData\LLMAnon\Batch1_CSA_Timestamps\text_not_found_Batch1.txt" \
    --text_not_found_batch2 "C:\LocalData\LLMAnon\Batch2_Focused_Timestamps\text_not_found_Batch2.txt" \
    --output "C:\LocalData\LLMAnon\Round2_Transcripts" \
    --model llama3.2
```

## Prerequisites

1. **Ollama must be running**: `ollama serve`
2. **Model installed**: `ollama pull llama3.2` (or your preferred model)

## What to Expect

### Input Processing
- **CSV**: ~3,340 total items
- **Text-not-found**: ~2,990 items (skipped automatically)
- **Validated PHI**: ~60 items to process

### Output Structure
```
C:\LocalData\LLMAnon\Round2_Transcripts\
├── [preserves nested folder structure from source]
│   ├── participant_LLMAnon_labeled.txt
│   └── participant_LLMAnon_resynth.txt
└── Round2_Processing_Summary.json
```

## Expected Statistics

Based on your audio work:
- **Transcript pairs found**: ~48 (Batch 1: 41, Batch 2: 7)
- **Labeled redactions**: ~60 successful replacements
- **Resynth redactions**: ~55-60 (some Ollama failures expected)
- **Text not in transcript**: ~5-10 (version mismatches between JSON and transcript)

## Validation Steps

After processing, check:

1. **Summary file**: Review `Round2_Processing_Summary.json`
2. **Spot check**: Open a few labeled and resynth files
3. **Search for PHI**: Verify original PHI text is gone
4. **CliniDeID tags**: Confirm `[*** NAME ***]` etc. are preserved
5. **LLMAnon markers**: Look for `[LLMAnon-Redacted]` in labeled version

## Key Features

✅ **Substring matching**: Finds files even with varied naming
✅ **Skips false positives**: Uses text-not-found logs automatically
✅ **Preserves structure**: Maintains nested folder organization
✅ **Dual output**: Both simple and contextual redactions
✅ **Ollama fallback**: Uses `[LLMAnon-Redacted-Fallback]` if generation fails
✅ **CliniDeID safe**: Preserves existing redaction markers

## Troubleshooting

### "Ollama not accessible"
- Start Ollama: `ollama serve`
- Check it's running: `ollama list`

### "Model not found"
- Install model: `ollama pull llama3.2`
- Or specify different model: `--model llama3`

### "Transcript not found"
- Expected for some items (JSON/transcript version mismatches)
- Check the summary to see how many succeeded

### "Text not in transcript"
- Expected for ~5-10 items
- Transcripts may differ slightly from JSON source
- Not an error, just logged for transparency

## Processing Time

- **~60 PHI items** × **~48 transcript pairs** × **2 versions** = ~120 operations
- With Ollama: ~5-10 minutes (0.5s delay per item for rate limiting)
- Without rate limiting: ~2-3 minutes

## Success Criteria

✅ ~48 transcript pairs processed
✅ ~60 labeled redactions applied
✅ ~55+ resynthesis redactions applied
✅ <5 Ollama failures (fallback used)
✅ Folder structure preserved
✅ Summary JSON generated

---

**Questions?** Check the processing summary JSON for detailed statistics.
