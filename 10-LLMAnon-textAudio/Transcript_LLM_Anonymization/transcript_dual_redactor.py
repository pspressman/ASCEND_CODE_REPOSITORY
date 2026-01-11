#!/usr/bin/env python3
"""
Dual Transcript Redaction Pipeline

Takes human-reviewed CSV from IT_transcript_analyzer and produces TWO versions:
1. Label-redacted: Replaces flagged text with [LLM-REDACTION]
2. Resynthesized: Uses Ollama to replace with semantically similar but fabricated content

Usage:
    python transcript_dual_redactor.py transcript_review.csv transcripts/ --output redacted_output/
    
Input CSV format (from infotheory_csv_exporter.py):
    REVIEW, FILE, RISK, CATEGORY, TEXT, REASONING, SUGGESTED_REPLACEMENT
    
The script only processes rows where REVIEW column = 'Y' (confirmed identifying)
"""

import csv
import requests
import json
import argparse
from pathlib import Path
from datetime import datetime
import time
from typing import Dict, List, Tuple
import re


class DualTranscriptRedactor:
    
    def __init__(self, ollama_url="http://localhost:11434", model="llama3.2"):
        """
        Initialize redactor with Ollama connection
        
        Args:
            ollama_url: URL for Ollama API
            model: Model name (default: llama3.2)
        """
        self.ollama_url = ollama_url
        self.model = model
        self.verify_connection()
        
        # Track statistics
        self.stats = {
            'files_processed': 0,
            'total_redactions': 0,
            'label_redactions': 0,
            'resynthesized_redactions': 0,
            'llm_failures': 0
        }
    
    def verify_connection(self):
        """Check Ollama connection and model availability"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            models = [m['name'] for m in response.json().get('models', [])]
            
            # Check for exact match or version match
            model_available = False
            for m in models:
                if self.model in m or m.startswith(self.model):
                    model_available = True
                    self.model = m  # Use exact name from server
                    break
            
            if not model_available:
                print(f"⚠️  Model {self.model} not found")
                print(f"\nTo install, run:")
                print(f"  ollama pull {self.model}")
                print(f"\nAvailable models on your system:")
                for m in models[:10]:
                    print(f"  - {m}")
                exit(1)
            
            print(f"✅ Connected to Ollama")
            print(f"🤖 Using model: {self.model}")
            print()
        except Exception as e:
            print(f"❌ Ollama not running or not accessible")
            print(f"   Error: {e}")
            print(f"\nStart Ollama with: ollama serve")
            exit(1)
    
    def create_resynthesis_prompt(self, original_text: str, context_before: str = "", 
                                   context_after: str = "") -> str:
        """
        Create prompt for LLM to resynthesize identifying content
        
        Args:
            original_text: The identifying text to replace
            context_before: Text before the identifying segment
            context_after: Text after the identifying segment
        """
        return f"""You are helping to de-identify clinical research transcripts while preserving ALL linguistic features for downstream NLP analysis. This text was flagged as HIGH SPECIFICITY identifying information.

ORIGINAL IDENTIFYING TEXT:
"{original_text}"

CONTEXT:
Before: "{context_before[-200:]}" 
After: "{context_after[:200]}"

CRITICAL: Your replacement must preserve these LINGUISTIC FEATURES for NLP analysis validity:

1. LEXICAL FEATURES:
   - Match word count (±2 words)
   - Match vocabulary diversity (if original varied words, vary yours; if repetitive, match that)
   - Keep any filled pauses ("um", "uh", "like") at same frequency

2. SYNTACTIC/POS FEATURES:
   - Match verb/noun/adjective/adverb ratios
   - Match pronoun types and frequency (1st/2nd/3rd person - if "I" used twice, use "I" twice)
   - Match determiner, conjunction, preposition frequencies
   - Match modal verb usage (can/could/would/should)
   - Preserve content density (ratio of content words to function words)

3. SENTIMENT:
   - Match emotional tone (negative → negative, neutral → neutral, positive → positive)
   - Preserve sentence-level polarity

4. NAMED ENTITIES:
   - Match entity type counts (if 2 person names + 1 location → generate 2 person names + 1 location)
   - Use fabricated but realistic entity names

5. SPECIFICITY LEVEL (CRITICAL):
   - This text was flagged as HIGH SPECIFICITY, so replace with DIFFERENT high specificity
   - Exact numbers → different exact numbers (e.g., "35 years" → "28 years")
   - Specific roles/titles → different specific roles ("head nurse" → "chemistry teacher")
   - Verifiable achievements → different verifiable achievements ("won award" → "directed program")
   - Exact dates → different exact dates
   - Specific locations → different specific locations
   - Safety: If original is vague (rare), keep replacement vague

6. SEMANTIC CATEGORY:
   - Medical profession → different profession (not necessarily medical)
   - Achievement → different achievement
   - Duration → different duration
   - Geographic reference → different geographic reference

7. COHERENCE (natural):
   - Maintain narrative flow with context
   - Keep grammatical structure similar

EXAMPLES:
- "I was head nurse there for 35 years and won the excellence award" →
  "I taught high school chemistry for 28 years and ran the science fair"
  (Matches: 1st person pronoun×1, past tense verbs×3, specific role, exact number, specific achievement, conjunction "and", similar length)

- "worked at the regional office since 1987" →
  "volunteered with the county museum from 1992"
  (Matches: past tense, specific location type, exact date, preposition structure)

- "everyone in the department knew I was struggling with um the new protocols" →
  "all my classmates saw I was confused by uh the assignment changes"
  (Matches: filled pause "uh", 1st person, past progressive, similar structure, sentiment)

OUTPUT FORMAT:
Return ONLY the replacement text. No quotes, no explanation, no preamble.

REPLACEMENT:"""

    def call_ollama(self, prompt: str, max_retries: int = 3) -> str:
        """
        Call Ollama API with retry logic
        
        Args:
            prompt: The prompt to send
            max_retries: Number of retry attempts
            
        Returns:
            Generated text or empty string on failure
        """
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.8,  # Some creativity for variety
                            "top_p": 0.9,
                            "num_predict": 100  # Keep responses concise
                        }
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    generated = result.get('response', '').strip()
                    
                    # Clean up common artifacts
                    generated = generated.strip('"\'')
                    generated = generated.split('\n')[0]  # Take first line only
                    
                    return generated
                else:
                    print(f"⚠️  API error: {response.status_code}")
                    
            except requests.exceptions.Timeout:
                print(f"⚠️  Timeout on attempt {attempt + 1}/{max_retries}")
                time.sleep(2)
            except Exception as e:
                print(f"⚠️  Error calling Ollama: {e}")
                time.sleep(2)
        
        self.stats['llm_failures'] += 1
        return ""
    
    def load_reviewed_csv(self, csv_path: Path) -> Dict[str, List[Dict]]:
        """
        Load human-reviewed CSV and organize by filename
        
        Args:
            csv_path: Path to the reviewed CSV file
            
        Returns:
            Dictionary mapping filename to list of confirmed redactions
        """
        redactions_by_file = {}
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Only process rows marked 'Y' (confirmed identifying)
                if row.get('REVIEW', '').strip().upper() != 'Y':
                    continue
                
                filename = row['FILE']
                
                if filename not in redactions_by_file:
                    redactions_by_file[filename] = []
                
                redactions_by_file[filename].append({
                    'text': row['TEXT'],
                    'risk': row['RISK'],
                    'reasoning': row['REASONING'],
                    'suggested_replacement': row.get('SUGGESTED_REPLACEMENT', '')
                })
        
        print(f"📋 Loaded redactions for {len(redactions_by_file)} files")
        total_redactions = sum(len(v) for v in redactions_by_file.values())
        print(f"   Total confirmed redactions: {total_redactions}")
        print()
        
        return redactions_by_file
    
    def find_context(self, transcript: str, target_text: str, 
                     context_window: int = 200) -> Tuple[str, str, int]:
        """
        Find target text in transcript and extract surrounding context
        
        Args:
            transcript: Full transcript text
            target_text: Text to find
            context_window: Characters of context to extract
            
        Returns:
            (context_before, context_after, position) or empty strings if not found
        """
        # Try exact match first
        pos = transcript.find(target_text)
        
        # If not found, try case-insensitive
        if pos == -1:
            lower_transcript = transcript.lower()
            lower_target = target_text.lower()
            pos = lower_transcript.find(lower_target)
            
            if pos != -1:
                # Extract actual text from transcript (preserving case)
                target_text = transcript[pos:pos + len(target_text)]
        
        # If still not found, try fuzzy matching (remove extra whitespace)
        if pos == -1:
            normalized_transcript = ' '.join(transcript.split())
            normalized_target = ' '.join(target_text.split())
            pos = normalized_transcript.find(normalized_target)
            
            if pos != -1:
                # Map back to original position (approximate)
                pos = transcript.find(normalized_target[:20])
        
        if pos == -1:
            return "", "", -1
        
        start = max(0, pos - context_window)
        end = min(len(transcript), pos + len(target_text) + context_window)
        
        context_before = transcript[start:pos]
        context_after = transcript[pos + len(target_text):end]
        
        return context_before, context_after, pos
    
    def process_transcript(self, transcript_path: Path, redactions: List[Dict],
                          output_dir: Path) -> Tuple[str, str]:
        """
        Process a single transcript and generate both redacted versions
        
        Args:
            transcript_path: Path to original transcript
            redactions: List of redaction dictionaries
            output_dir: Output directory
            
        Returns:
            Tuple of (label_redacted_text, resynthesized_text)
        """
        print(f"📝 Processing: {transcript_path.name}")
        print(f"   Redactions to apply: {len(redactions)}")
        
        # Read original transcript
        with open(transcript_path, 'r', encoding='utf-8') as f:
            original = f.read()
        
        # Initialize both versions
        label_redacted = original
        resynthesized = original
        
        # Sort redactions by position (process from end to start to preserve positions)
        redactions_with_pos = []
        for redaction in redactions:
            text = redaction['text']
            _, _, pos = self.find_context(original, text)
            
            if pos != -1:
                redactions_with_pos.append((pos, text, redaction))
            else:
                print(f"   ⚠️  Could not locate text: {text[:50]}...")
        
        # Sort by position (descending) to process from end to start
        redactions_with_pos.sort(key=lambda x: x[0], reverse=True)
        
        successful_redactions = 0
        
        for pos, text, redaction in redactions_with_pos:
            # Apply label redaction
            label_redacted = (
                label_redacted[:pos] + 
                "[LLM-REDACTION]" + 
                label_redacted[pos + len(text):]
            )
            
            # Generate resynthesized replacement
            context_before, context_after, _ = self.find_context(original, text)
            
            prompt = self.create_resynthesis_prompt(text, context_before, context_after)
            replacement = self.call_ollama(prompt)
            
            if replacement:
                resynthesized = (
                    resynthesized[:pos] + 
                    replacement + 
                    resynthesized[pos + len(text):]
                )
                successful_redactions += 1
                print(f"   ✓ Resynthesized: {text[:50]}... → {replacement[:50]}...")
            else:
                # Fallback to label redaction if LLM fails
                resynthesized = (
                    resynthesized[:pos] + 
                    "[LLM-REDACTION-FALLBACK]" + 
                    resynthesized[pos + len(text):]
                )
                print(f"   ⚠️  Fallback redaction for: {text[:50]}...")
        
        self.stats['total_redactions'] += len(redactions_with_pos)
        self.stats['label_redactions'] += len(redactions_with_pos)
        self.stats['resynthesized_redactions'] += successful_redactions
        
        print(f"   ✅ Complete: {successful_redactions}/{len(redactions_with_pos)} resynthesized")
        print()
        
        return label_redacted, resynthesized
    
    def save_redacted_transcripts(self, filename: str, label_version: str,
                                  resynth_version: str, output_dir: Path):
        """
        Save both redacted versions with clear naming
        
        Args:
            filename: Original filename
            label_version: Label-redacted text
            resynth_version: Resynthesized text
            output_dir: Output directory
        """
        stem = Path(filename).stem
        
        # Create subdirectories
        label_dir = output_dir / "label_redacted"
        resynth_dir = output_dir / "resynthesized"
        label_dir.mkdir(parents=True, exist_ok=True)
        resynth_dir.mkdir(parents=True, exist_ok=True)
        
        # Save label version
        label_path = label_dir / f"{stem}_LABEL_REDACTED.txt"
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write(label_version)
        
        # Save resynthesized version
        resynth_path = resynth_dir / f"{stem}_RESYNTHESIZED.txt"
        with open(resynth_path, 'w', encoding='utf-8') as f:
            f.write(resynth_version)
        
        print(f"   💾 Saved: {label_path.name}")
        print(f"   💾 Saved: {resynth_path.name}")
    
    def process_batch(self, csv_path: Path, transcript_dir: Path, output_dir: Path):
        """
        Process all transcripts based on reviewed CSV
        
        Args:
            csv_path: Path to reviewed CSV
            transcript_dir: Directory containing original transcripts
            output_dir: Output directory for redacted versions
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load redactions
        redactions_by_file = self.load_reviewed_csv(csv_path)
        
        if not redactions_by_file:
            print("⚠️  No confirmed redactions found in CSV (REVIEW column must be 'Y')")
            return
        
        # Process each file
        transcript_dir = Path(transcript_dir)
        
        for filename, redactions in redactions_by_file.items():
            # Find matching transcript file
            matching_files = list(transcript_dir.glob(f"{Path(filename).stem}*"))
            
            if not matching_files:
                print(f"⚠️  Transcript not found for: {filename}")
                continue
            
            transcript_path = matching_files[0]
            
            # Process transcript
            label_version, resynth_version = self.process_transcript(
                transcript_path, redactions, output_dir
            )
            
            # Save both versions
            self.save_redacted_transcripts(
                filename, label_version, resynth_version, output_dir
            )
            
            self.stats['files_processed'] += 1
        
        # Generate summary report
        self.generate_summary_report(output_dir)
    
    def generate_summary_report(self, output_dir: Path):
        """Generate summary report of redaction process"""
        
        report_lines = [
            "# Dual Transcript Redaction Summary",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Model Used:** {self.model}",
            "",
            "## Processing Statistics",
            "",
            f"- **Files Processed:** {self.stats['files_processed']}",
            f"- **Total Redactions Applied:** {self.stats['total_redactions']}",
            f"- **Successful Resynthesizations:** {self.stats['resynthesized_redactions']}",
            f"- **LLM Fallbacks:** {self.stats['llm_failures']}",
            "",
            "## Output Structure",
            "",
            "```",
            "output_directory/",
            "├── label_redacted/",
            "│   └── *_LABEL_REDACTED.txt",
            "├── resynthesized/",
            "│   └── *_RESYNTHESIZED.txt",
            "└── REDACTION_SUMMARY.md (this file)",
            "```",
            "",
            "## Redaction Methods",
            "",
            "### Label Redacted",
            "- Simple replacement with `[LLM-REDACTION]`",
            "- Safe for legal/compliance review",
            "- May disrupt NLP analysis due to artificial markers",
            "",
            "### Resynthesized",
            "- LLM-generated semantically similar replacements",
            "- Preserves narrative structure for NLP analysis",
            "- Completely fabricated and untraceable",
            "- Fallback to `[LLM-REDACTION-FALLBACK]` if generation fails",
            "",
            "## Quality Assurance",
            "",
            "**Recommended QA Steps:**",
            "1. Manually review sample of resynthesized transcripts",
            "2. Verify semantic coherence maintained",
            "3. Confirm no identifying details remain",
            "4. Test with downstream NLP pipelines",
            "",
            "## Next Steps",
            "",
            "1. ✅ Audio redaction at identified timestamps (already built)",
            "2. 🔍 Quality assurance review of resynthesized content",
            "3. 📊 Run NLP analyses on resynthesized transcripts",
            "4. 📝 Document any issues for future improvement",
            "",
            "---",
            f"*Generated by Dual Transcript Redactor v1.0*"
        ]
        
        report_path = output_dir / 'REDACTION_SUMMARY.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print("\n" + "="*60)
        print("📊 PROCESSING COMPLETE")
        print("="*60)
        print(f"\n✅ Files processed: {self.stats['files_processed']}")
        print(f"📝 Total redactions: {self.stats['total_redactions']}")
        print(f"🎨 Successful resynthesizations: {self.stats['resynthesized_redactions']}")
        if self.stats['llm_failures'] > 0:
            print(f"⚠️  LLM fallbacks: {self.stats['llm_failures']}")
        print(f"\n📄 Summary report: {report_path}")
        print(f"📁 Label redacted: {output_dir / 'label_redacted'}")
        print(f"📁 Resynthesized: {output_dir / 'resynthesized'}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Dual Transcript Redaction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python transcript_dual_redactor.py review.csv transcripts/ --output redacted_output/
  
  # With custom model
  python transcript_dual_redactor.py review.csv transcripts/ --output redacted/ --model llama3.2:3b
  
  # With custom Ollama URL
  python transcript_dual_redactor.py review.csv transcripts/ --output redacted/ --ollama-url http://192.168.1.100:11434

Input CSV Format:
  The CSV must be from infotheory_csv_exporter.py and must have a REVIEW column
  where 'Y' marks confirmed identifying content.
  
Outputs:
  - label_redacted/*_LABEL_REDACTED.txt: Simple [LLM-REDACTION] replacements
  - resynthesized/*_RESYNTHESIZED.txt: LLM-generated semantic replacements
        """
    )
    
    parser.add_argument('csv_file', type=Path,
                       help='Reviewed CSV file from infotheory_csv_exporter.py')
    parser.add_argument('transcript_dir', type=Path,
                       help='Directory containing original transcript files')
    parser.add_argument('--output', type=Path, default=Path('redacted_output'),
                       help='Output directory for redacted transcripts')
    parser.add_argument('--model', type=str, default='llama3.2',
                       help='Ollama model name (default: llama3.2)')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434',
                       help='Ollama API URL (default: http://localhost:11434)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.csv_file.exists():
        print(f"❌ CSV file not found: {args.csv_file}")
        exit(1)
    
    if not args.transcript_dir.exists():
        print(f"❌ Transcript directory not found: {args.transcript_dir}")
        exit(1)
    
    print("="*60)
    print("DUAL TRANSCRIPT REDACTION PIPELINE")
    print("="*60)
    print()
    
    # Initialize redactor
    redactor = DualTranscriptRedactor(
        ollama_url=args.ollama_url,
        model=args.model
    )
    
    # Process batch
    redactor.process_batch(args.csv_file, args.transcript_dir, args.output)


if __name__ == "__main__":
    main()
