#!/usr/bin/env python3
"""
Find and remove duplicate files from a local directory.
Identifies files like "file.ext" and "file 2.ext", verifies they're identical
using file size and MD5 hash, then removes duplicates.
"""

import os
import re
import hashlib
from collections import defaultdict
from pathlib import Path

def compute_file_hash(filepath, chunk_size=8192):
    """Compute MD5 hash of a file."""
    try:
        md5 = hashlib.md5()
        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                md5.update(chunk)
        return md5.hexdigest()
    except Exception as e:
        print(f"    ⚠️  Error hashing {filepath}: {e}")
        return None

def find_duplicate_groups(directory):
    """
    Group files that might be duplicates based on naming pattern.
    E.g., "file.txt" and "file 2.txt" would be in the same group.
    """
    # Pattern to match numbered duplicates like " 2", " 3", etc.
    duplicate_pattern = re.compile(r'^(.+?)( \d+)?(\.[^.]+)?$')
    
    groups = defaultdict(list)
    
    # Walk through directory
    for root, dirs, files in os.walk(directory):
        for filename in files:
            # Skip hidden files
            if filename.startswith('.'):
                continue
            
            filepath = os.path.join(root, filename)
            filesize = os.path.getsize(filepath)
            
            match = duplicate_pattern.match(filename)
            if match:
                base_name = match.group(1)
                extension = match.group(3) if match.group(3) else ''
                number = match.group(2)
                
                # Create a group key from base name and extension
                group_key = f"{base_name}{extension}"
                groups[group_key].append({
                    'name': filename,
                    'path': filepath,
                    'size': filesize,
                    'number': number if number else '',
                    'sort_key': 0 if not number else int(number.strip())
                })
    
    # Filter to only groups with multiple files
    duplicate_groups = {k: v for k, v in groups.items() if len(v) > 1}
    
    return duplicate_groups

def format_size(size):
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"

def verify_and_remove_duplicates(duplicate_groups, dry_run=True):
    """
    Verify files are identical and remove duplicates.
    Keeps the file without a number (the "original").
    """
    results = {
        'verified_duplicates': [],
        'different_files': [],
        'errors': [],
        'bytes_saved': 0
    }
    
    for group_name, files in duplicate_groups.items():
        print(f"\n{'='*80}")
        print(f"Checking group: {group_name}")
        print(f"Files in group: {len(files)}")
        
        # Sort by number (original file first)
        files_sorted = sorted(files, key=lambda x: x['sort_key'])
        
        # Display all files in group
        for f in files_sorted:
            print(f"  - {f['name']}")
            print(f"    Size: {format_size(f['size'])} ({f['size']:,} bytes)")
            print(f"    Path: {f['path']}")
        
        # Compare all files by size first
        sizes = [f['size'] for f in files_sorted]
        if len(set(sizes)) > 1:
            print("\n  ⚠️  FILES HAVE DIFFERENT SIZES - NOT DUPLICATES")
            results['different_files'].append(group_name)
            continue
        
        print(f"\n  ✓ All files have same size: {format_size(sizes[0])}")
        
        # Compute hashes for all files
        print("  Computing file hashes...")
        file_hashes = []
        for f in files_sorted:
            hash_val = compute_file_hash(f['path'])
            if hash_val:
                file_hashes.append(hash_val)
                print(f"    {f['name']}: {hash_val}")
            else:
                results['errors'].append(f['name'])
        
        # Check if all hashes match
        if len(file_hashes) != len(files_sorted):
            print("  ⚠️  Could not hash all files - skipping")
            continue
        
        if len(set(file_hashes)) > 1:
            print("\n  ⚠️  FILES HAVE DIFFERENT CONTENT - NOT DUPLICATES")
            results['different_files'].append(group_name)
            continue
        
        print("\n  ✓ All files have identical content!")
        
        # Keep the first file (original without number), delete the rest
        original = files_sorted[0]
        duplicates = files_sorted[1:]
        
        print(f"\n  📌 Keeping: {original['name']}")
        print(f"     Path: {original['path']}")
        
        for dup in duplicates:
            dup_name = dup['name']
            dup_path = dup['path']
            
            if dry_run:
                print(f"\n  🗑️  Would delete: {dup_name}")
                print(f"     Path: {dup_path}")
                print(f"     Would save: {format_size(dup['size'])}")
                results['bytes_saved'] += dup['size']
            else:
                try:
                    os.remove(dup_path)
                    print(f"\n  ✅ Deleted: {dup_name}")
                    print(f"     Path: {dup_path}")
                    print(f"     Freed: {format_size(dup['size'])}")
                    results['verified_duplicates'].append(dup_name)
                    results['bytes_saved'] += dup['size']
                except Exception as e:
                    print(f"\n  ❌ Error deleting {dup_name}: {e}")
                    results['errors'].append(dup_name)
    
    return results

def main():
    print("="*80)
    print("Local Duplicate File Finder and Remover")
    print("="*80)
    
    # Get directory from user
    print("\nEnter the full path to the directory to scan:")
    print("(e.g., /path/to/volumes/MyDrive/CSAND_AUDIO/Paralang/WavsAndTranscripts-CSA-research)")
    directory = input("\nPath: ").strip()
    
    # Remove quotes if user pasted with quotes
    directory = directory.strip('"').strip("'")
    
    # Expand ~ to home directory
    directory = os.path.expanduser(directory)
    
    if not os.path.exists(directory):
        print(f"\n❌ ERROR: Directory not found: {directory}")
        return
    
    if not os.path.isdir(directory):
        print(f"\n❌ ERROR: Path is not a directory: {directory}")
        return
    
    print(f"\n✓ Found directory: {directory}")
    
    # Count files
    file_count = sum(len(files) for _, _, files in os.walk(directory))
    print(f"  Total files: {file_count:,}")
    
    # Find potential duplicate groups
    print("\nAnalyzing files for duplicates...")
    duplicate_groups = find_duplicate_groups(directory)
    
    if not duplicate_groups:
        print("\n✅ No duplicate files found!")
        return
    
    print(f"\n🔍 Found {len(duplicate_groups)} potential duplicate groups")
    
    # First pass: DRY RUN
    print("\n" + "="*80)
    print("DRY RUN - No files will be deleted")
    print("="*80)
    
    results = verify_and_remove_duplicates(duplicate_groups, dry_run=True)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total duplicate groups checked: {len(duplicate_groups)}")
    print(f"Different files (not duplicates): {len(results['different_files'])}")
    print(f"Errors: {len(results['errors'])}")
    
    # Calculate how many would be deleted
    total_duplicates = sum(len(files) - 1 for files in duplicate_groups.values() 
                          if len(files) > 1)
    verified_duplicates = len([f for group in duplicate_groups.values() 
                               for f in group[1:]])
    
    print(f"\nPotential duplicates that can be deleted: {verified_duplicates}")
    print(f"Space that would be freed: {format_size(results['bytes_saved'])}")
    
    # Ask for confirmation to actually delete
    print("\n" + "="*80)
    response = input("\nDo you want to DELETE the duplicate files? (type 'yes' to confirm): ").strip().lower()
    
    if response == 'yes':
        print("\n🗑️  DELETING DUPLICATES...")
        results = verify_and_remove_duplicates(duplicate_groups, dry_run=False)
        print(f"\n✅ Deleted {len(results['verified_duplicates'])} duplicate files")
        print(f"✅ Freed {format_size(results['bytes_saved'])} of disk space")
    else:
        print("\n❌ Deletion cancelled. No files were deleted.")

if __name__ == "__main__":
    main()
