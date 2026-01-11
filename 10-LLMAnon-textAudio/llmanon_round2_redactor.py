#!/usr/bin/env python3
"""
LLMAnon Round 2 Transcript Redaction
Applies validated LLMAnon PHI redactions to CliniDeID transcripts.

Processes BOTH CliniDeID versions:
1. .deid.piiCategoryTag → adds [LLMAnon-Redacted]
2. .deid.resynthesis → adds Ollama replacements

Usage:
    python llmanon_round2_redactor.py \\
        --csv "C:\path\to\data\LLMAnon\LLM-Anon-reviewed.csv" \\
        --transcripts "C:\path\to\data\ASCEND_PHI\DeID\CliniDeID_organized copy" \\
        --text_not_found_batch1 "C:\path\to\data\LLMAnon\Batch1_CSA_Timestamps\text_not_found_Batch1.txt" \\
        --text_not_found_batch2 "C:\path\to\data\LLMAnon\Batch2_Focused_Timestamps\text_not_found_Batch2.txt" \\
        --output "C:\path\to\data\LLMAnon\Round2_Transcripts"
"""

import argparse
import csv
import requests
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict


class LLMAnon_Round2_Redactor:
    
    def __init__(self, ollama_url="http://localhost:11434", model="llama3.2"):
        """Initialize with Ollama connection"""
        self.ollama_url = ollama_url
        self.model = model
        self.verify_ollama()
        
        # Statistics
        self.stats = {
            'csv_total': 0,
            'text_not_found': 0,
            'validated_phi': 0,
            'transcripts_found': 0,
            'transcripts_not_found': 0,
            'labeled_redactions': 0,
            'resynth_redactions': 0,
            'ollama_failures': 0,
            'text_not_in_transcript': 0
        }
        
    def verify_ollama(self):
        """Check Ollama connection"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            models = [m['name'] for m in response.json().get('models', [])]
            
            model_found = False
            for m in models:
                if self.model in m or m.startswith(self.model):
                    self.model = m
                    model_found = True
                    break
            
            if not model_found:
                print(f"⚠️  Model {self.model} not found. Available models:")
                for m in models[:5]:
                    print(f"    - {m}")
                print(f"\nTo install: ollama pull {self.model}")
                exit(1)
            
            print(f"✅ Ollama connected: {self.model}")
        except Exception as e:
            print(f"❌ Ollama not accessible: {e}")
            print("   Start with: ollama serve")
            exit(1)
    
    def load_text_not_found(self, path: Path) -> Set[Tuple[str, str]]:
        """
        Load text-not-found log and return set of (participant_id, text) tuples
        
        Log format: "Row X: 'TEXT' not found in PARTICIPANT_ID"
        """
        not_found = set()
        
        if not path.exists():
            print(f"⚠️  Text-not-found log not found: {path}")
            return not_found
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Parse: "Row X: 'TEXT' not found in PARTICIPANT_ID"
                if "not found in" in line and "Row" in line:
                    try:
                        # Split on "not found in"
                        parts = line.split("not found in")
                        if len(parts) == 2:
                            # Extract text from 'TEXT' part (between quotes)
                            text_part = parts[0]
                            if "'" in text_part:
                                text = text_part.split("'")[1]
                            else:
                                continue
                            
                            # Extract participant ID (after "not found in")
                            participant_id = parts[1].strip()
                            
                            not_found.add((participant_id, text))
                    except Exception as e:
                        # Skip malformed lines
                        continue
        
        print(f"   Loaded {len(not_found)} text-not-found entries from {path.name}")
        return not_found
    
    def load_validated_phi(self, csv_path: Path, text_not_found_batch1: Path, 
                          text_not_found_batch2: Path) -> List[Dict]:
        """
        Load LLMAnon CSV and filter to only validated PHI
        (exclude items in text-not-found logs)
        """
        print("\n" + "="*60)
        print("LOADING VALIDATED PHI")
        print("="*60)
        
        # Load text-not-found logs
        print("\nLoading text-not-found logs...")
        not_found_batch1 = self.load_text_not_found(text_not_found_batch1)
        not_found_batch2 = self.load_text_not_found(text_not_found_batch2)
        all_not_found = not_found_batch1 | not_found_batch2
        
        print(f"\nTotal text-not-found entries: {len(all_not_found)}")
        
        # Load CSV
        print(f"\nLoading LLMAnon CSV: {csv_path.name}")
        validated_phi = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                self.stats['csv_total'] += 1
                
                participant_id = row.get('FILE-basename', '').strip()
                text = row.get('TEXT', '').strip()
                
                # Skip if in text-not-found
                if (participant_id, text) in all_not_found:
                    self.stats['text_not_found'] += 1
                    continue
                
                # Skip empty entries
                if not participant_id or not text:
                    continue
                
                validated_phi.append({
                    'participant_id': participant_id,
                    'text': text,
                    'category': row.get('CATEGORY', ''),
                    'risk': row.get('RISK', ''),
                    'suggested_replacement': row.get('SUGGESTED_REPLACEMENT', '')
                })
        
        self.stats['validated_phi'] = len(validated_phi)
        
        print(f"\n📊 CSV Summary:")
        print(f"   Total items in CSV: {self.stats['csv_total']}")
        print(f"   Text-not-found (skipped): {self.stats['text_not_found']}")
        print(f"   Validated PHI to process: {self.stats['validated_phi']}")
        
        return validated_phi
    
    def build_transcript_index(self, transcript_dir: Path) -> Dict[str, List[Path]]:
        """
        Build index of all transcript files for substring matching
        
        Returns dict: {basename: [list of matching transcript paths]}
        """
        print("\n" + "="*60)
        print("BUILDING TRANSCRIPT INDEX")
        print("="*60)
        
        transcript_index = defaultdict(list)
        
        # Find all .deid.piiCategoryTag.txt and .deid.resynthesis.txt files
        labeled_files = list(transcript_dir.rglob('*.deid.piiCategoryTag.txt'))
        resynth_files = list(transcript_dir.rglob('*.deid.resynthesis.txt'))
        
        all_files = labeled_files + resynth_files
        
        print(f"\nFound {len(labeled_files)} labeled transcripts")
        print(f"Found {len(resynth_files)} resynthesis transcripts")
        print(f"Total: {len(all_files)} transcript files")
        
        # Build index - group by pairs
        for file_path in all_files:
            # Store the file path for any substring search
            transcript_index[str(file_path)] = [file_path]
        
        return transcript_index
    
    def find_transcript_pair(self, participant_id: str, 
                            transcript_dir: Path) -> Optional[Tuple[Path, Path]]:
        """
        Find matching transcript pair using substring search
        
        Returns: (labeled_path, resynth_path) or None
        """
        # Search for files containing the participant_id
        labeled_matches = list(transcript_dir.rglob(f'*{participant_id}*.deid.piiCategoryTag.txt'))
        resynth_matches = list(transcript_dir.rglob(f'*{participant_id}*.deid.resynthesis.txt'))
        
        if labeled_matches and resynth_matches:
            # Return first match (should be only one due to consistent naming)
            return (labeled_matches[0], resynth_matches[0])
        
        return None
    
    def call_ollama(self, text: str, phi_word: str, category: str, 
                    risk: str, suggested: str) -> Optional[str]:
        """
        Call Ollama to generate replacement text
        
        Keeps it simple - just ask for a semantically similar but fabricated replacement
        """
        prompt = f"""You are de-identifying clinical transcripts. Replace the following identifying information with semantically similar but completely fabricated content.

IDENTIFYING TEXT TO REPLACE: "{phi_word}"
CATEGORY: {category}
RISK LEVEL: {risk}
SUGGESTED REPLACEMENT: {suggested}

Requirements:
- Match the length (within 1-2 words)
- Match the specificity level (if specific, keep specific; if vague, keep vague)
- Keep similar grammatical structure
- Make it completely fabricated and untraceable
- Return ONLY the replacement text, no quotes or explanation

REPLACEMENT:"""

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.8,
                        "top_p": 0.9,
                        "num_predict": 50
                    }
                },
                timeout=120  # Increased for 70B model
            )
            
            if response.status_code == 200:
                result = response.json()
                generated = result.get('response', '').strip()
                
                # Clean up
                generated = generated.strip('"\'')
                generated = generated.split('\n')[0]
                
                return generated if generated else None
            
        except Exception as e:
            print(f"      ⚠️  Ollama error: {e}")
            return None
        
        return None
    
    def redact_transcript(self, transcript_text: str, phi_word: str, 
                         mode: str, **kwargs) -> Tuple[str, bool]:
        """
        Redact PHI from transcript
        
        Args:
            transcript_text: Full transcript content
            phi_word: The PHI text to find and replace
            mode: 'labeled' or 'resynth'
            **kwargs: Additional args for Ollama (category, risk, etc.)
            
        Returns:
            (modified_text, success)
        """
        # Check if PHI text exists in transcript
        if phi_word not in transcript_text:
            return transcript_text, False
        
        if mode == 'labeled':
            # Simple replacement
            modified = transcript_text.replace(phi_word, '[LLMAnon-Redacted]')
            return modified, True
        
        elif mode == 'resynth':
            # Ollama replacement
            replacement = self.call_ollama(
                transcript_text, phi_word,
                kwargs.get('category', ''),
                kwargs.get('risk', ''),
                kwargs.get('suggested', '')
            )
            
            if replacement:
                modified = transcript_text.replace(phi_word, replacement)
                return modified, True
            else:
                # Fallback to label if Ollama fails
                modified = transcript_text.replace(phi_word, '[LLMAnon-Redacted-Fallback]')
                return modified, False
        
        return transcript_text, False
    
    def process_phi_item(self, phi_item: Dict, transcript_dir: Path, 
                        output_dir: Path):
        """Process a single validated PHI item"""
        
        participant_id = phi_item['participant_id']
        phi_text = phi_item['text']
        
        # Find transcript pair
        transcript_pair = self.find_transcript_pair(participant_id, transcript_dir)
        
        if not transcript_pair:
            self.stats['transcripts_not_found'] += 1
            return
        
        labeled_path, resynth_path = transcript_pair
        self.stats['transcripts_found'] += 1
        
        print(f"\n  📄 {participant_id}")
        print(f"     PHI: {phi_text[:50]}...")
        
        # Load both transcripts
        with open(labeled_path, 'r', encoding='utf-8') as f:
            labeled_text = f.read()
        
        with open(resynth_path, 'r', encoding='utf-8') as f:
            resynth_text = f.read()
        
        # Process labeled version
        labeled_modified, labeled_success = self.redact_transcript(
            labeled_text, phi_text, 'labeled'
        )
        
        if labeled_success:
            print(f"     ✅ Labeled redaction applied")
            self.stats['labeled_redactions'] += 1
        else:
            print(f"     ⚠️  Text not found in labeled transcript")
            self.stats['text_not_in_transcript'] += 1
        
        # Process resynth version
        resynth_modified, resynth_success = self.redact_transcript(
            resynth_text, phi_text, 'resynth',
            category=phi_item['category'],
            risk=phi_item['risk'],
            suggested=phi_item['suggested_replacement']
        )
        
        if resynth_success:
            print(f"     ✅ Ollama replacement applied")
            self.stats['resynth_redactions'] += 1
        else:
            if phi_text in resynth_text:
                print(f"     ⚠️  Ollama failed, used fallback")
                self.stats['ollama_failures'] += 1
            else:
                print(f"     ⚠️  Text not found in resynth transcript")
                self.stats['text_not_in_transcript'] += 1
        
        # Save outputs (preserve folder structure)
        relative_path = labeled_path.relative_to(transcript_dir)
        output_subdir = output_dir / relative_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filenames
        base_name = labeled_path.stem.replace('.deid.piiCategoryTag', '')
        labeled_output = output_subdir / f"{base_name}_LLMAnon_labeled.txt"
        resynth_output = output_subdir / f"{base_name}_LLMAnon_resynth.txt"
        
        # Write outputs
        with open(labeled_output, 'w', encoding='utf-8') as f:
            f.write(labeled_modified)
        
        with open(resynth_output, 'w', encoding='utf-8') as f:
            f.write(resynth_modified)
        
        print(f"     💾 Saved: {labeled_output.name}")
        print(f"     💾 Saved: {resynth_output.name}")
    
    def process_batch(self, csv_path: Path, transcript_dir: Path, 
                     text_not_found_batch1: Path, text_not_found_batch2: Path,
                     output_dir: Path):
        """Process all validated PHI items"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load validated PHI
        validated_phi = self.load_validated_phi(
            csv_path, text_not_found_batch1, text_not_found_batch2
        )
        
        if not validated_phi:
            print("\n❌ No validated PHI to process!")
            return
        
        # Build transcript index
        transcript_index = self.build_transcript_index(transcript_dir)
        
        # Process each PHI item
        print("\n" + "="*60)
        print("PROCESSING PHI ITEMS")
        print("="*60)
        
        for i, phi_item in enumerate(validated_phi, 1):
            print(f"\n[{i}/{len(validated_phi)}]", end='')
            self.process_phi_item(phi_item, transcript_dir, output_dir)
            
            # Rate limiting for Ollama (not needed with 70B - naturally slow)
            # time.sleep(0.5)
        
        # Generate summary
        self.generate_summary(output_dir)
    
    def generate_summary(self, output_dir: Path):
        """Generate processing summary"""
        
        summary = {
            'processing_timestamp': datetime.now().isoformat(),
            'statistics': self.stats,
            'model_used': self.model
        }
        
        summary_path = output_dir / 'Round2_Processing_Summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*60)
        print("ROUND 2 REDACTION COMPLETE")
        print("="*60)
        print(f"\n📊 Processing Statistics:")
        print(f"   CSV total items: {self.stats['csv_total']}")
        print(f"   Text-not-found (skipped): {self.stats['text_not_found']}")
        print(f"   Validated PHI processed: {self.stats['validated_phi']}")
        print(f"\n📄 Transcript Results:")
        print(f"   Transcript pairs found: {self.stats['transcripts_found']}")
        print(f"   Transcript pairs not found: {self.stats['transcripts_not_found']}")
        print(f"\n✏️  Redaction Results:")
        print(f"   Labeled redactions: {self.stats['labeled_redactions']}")
        print(f"   Resynth redactions: {self.stats['resynth_redactions']}")
        print(f"   Ollama failures (fallback used): {self.stats['ollama_failures']}")
        print(f"   Text not in transcript: {self.stats['text_not_in_transcript']}")
        print(f"\n📁 Output: {output_dir}")
        print(f"📄 Summary: {summary_path}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='LLMAnon Round 2: Apply validated PHI redactions to CliniDeID transcripts'
    )
    
    parser.add_argument('--csv', type=Path, required=True,
                       help='LLMAnon CSV (LLM-Anon-reviewed.csv)')
    parser.add_argument('--transcripts', type=Path, required=True,
                       help='Root directory of CliniDeID transcripts')
    parser.add_argument('--text_not_found_batch1', type=Path, required=True,
                       help='Text-not-found log for Batch 1')
    parser.add_argument('--text_not_found_batch2', type=Path, required=True,
                       help='Text-not-found log for Batch 2')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output directory for Round 2 transcripts')
    parser.add_argument('--model', type=str, default='llama3.2',
                       help='Ollama model name (default: llama3.2)')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434',
                       help='Ollama API URL')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.csv.exists():
        print(f"❌ CSV not found: {args.csv}")
        exit(1)
    
    if not args.transcripts.exists():
        print(f"❌ Transcript directory not found: {args.transcripts}")
        exit(1)
    
    print("="*60)
    print("LLMANON ROUND 2 TRANSCRIPT REDACTION")
    print("="*60)
    
    # Initialize redactor
    redactor = LLMAnon_Round2_Redactor(
        ollama_url=args.ollama_url,
        model=args.model
    )
    
    # Process batch
    redactor.process_batch(
        args.csv,
        args.transcripts,
        args.text_not_found_batch1,
        args.text_not_found_batch2,
        args.output
    )


if __name__ == "__main__":
    main()
