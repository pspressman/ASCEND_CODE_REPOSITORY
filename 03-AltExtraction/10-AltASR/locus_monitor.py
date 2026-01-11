#!/usr/bin/env python3
"""
Simple process monitor for Locus
Shows CPU/RAM usage and status of: Vosk, IT analyzer, NLP pipeline
"""

import psutil
import time
import os
from datetime import datetime

def get_process_info(search_terms):
    """Find processes matching any search term"""
    matches = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if any(term.lower() in cmdline.lower() for term in search_terms):
                matches.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'cpu': proc.info['cpu_percent'],
                    'mem_gb': proc.info['memory_info'].rss / (1024**3),
                    'cmd': cmdline[:80]
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return matches

def main():
    os.system('clear')
    
    while True:
        # Get system stats
        cpu_percent = psutil.cpu_percent(interval=1, percpu=False)
        mem = psutil.virtual_memory()
        
        # Get processes
        vosk_procs = get_process_info(['vosk', 'faster_whisper'])
        it_procs = get_process_info(['IT_transcript_analyzer', 'ollama'])
        nlp_procs = get_process_info(['coherence', 'nlp_pipeline'])
        
        # Display
        print(f"\n{'='*80}")
        print(f"LOCUS STATUS - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*80}")
        print(f"CPU: {cpu_percent:5.1f}% | RAM: {mem.used/1024**3:5.1f}/{mem.total/1024**3:5.1f} GB ({mem.percent:5.1f}%)")
        print(f"{'='*80}\n")
        
        # Vosk/ASR
        print(f"VOSK/ASR ({len(vosk_procs)} processes)")
        if vosk_procs:
            for p in vosk_procs:
                print(f"  PID {p['pid']:5d} | CPU {p['cpu']:5.1f}% | RAM {p['mem_gb']:5.2f}GB | {p['cmd']}")
        else:
            print(f"  [Not running]")
        print()
        
        # IT Analyzer
        print(f"IT ANALYZER / OLLAMA ({len(it_procs)} processes)")
        if it_procs:
            for p in it_procs:
                print(f"  PID {p['pid']:5d} | CPU {p['cpu']:5.1f}% | RAM {p['mem_gb']:5.2f}GB | {p['cmd']}")
        else:
            print(f"  [Not running]")
        print()
        
        # NLP Pipeline
        print(f"NLP COHERENCE ({len(nlp_procs)} processes)")
        if nlp_procs:
            for p in nlp_procs:
                print(f"  PID {p['pid']:5d} | CPU {p['cpu']:5.1f}% | RAM {p['mem_gb']:5.2f}GB | {p['cmd']}")
        else:
            print(f"  [Not running]")
        
        print(f"\n{'='*80}")
        print("Press Ctrl+C to exit | Refreshes every 5 seconds")
        
        time.sleep(5)
        os.system('clear')

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nMonitor stopped.")
