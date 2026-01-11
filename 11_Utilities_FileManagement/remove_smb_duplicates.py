#!/usr/bin/env python3
"""
Script to find and remove duplicate files on an SMB share.
Identifies files like "file.ext" and "file 2.ext", verifies they're identical
using file size and MD5 hash, then removes duplicates.
"""

import os
import re
import hashlib
from collections import defaultdict
from smb.SMBConnection import SMBConnection
from smb.smb_structs import OperationFailure
import tempfile

# SMB Share Configuration
SMB_SERVER = "10.0.0.49"
SMB_SHARE = "video_research"
SMB_PATH = "/CSAND_AUDIO/Paralang/WavsAndTranscripts-CSA-research"
SMB_USERNAME = ""  # Empty for guest/anonymous
SMB_PASSWORD = ""
SMB_DOMAIN = ""
SMB_CLIENT_NAME = "python-client"

def connect_to_smb():
    """Connect to the SMB share."""
    print(f"Connecting to SMB share: //{SMB_SERVER}/{SMB_SHARE}")
    
    # Try anonymous connection first
    conn = SMBConnection(
        SMB_USERNAME if SMB_USERNAME else "guest",
        SMB_PASSWORD,
        SMB_CLIENT_NAME,
        SMB_SERVER,
        domain=SMB_DOMAIN,
        use_ntlm_v2=True,
        is_direct_tcp=True
    )
    
    try:
        if conn.connect(SMB_SERVER, 445):
            print("Connected successfully!")
            return conn
        else:
            print("Failed to connect to SMB share")
            return None
    except Exception as e:
        print(f"Connection error: {e}")
        return None

def list_files(conn, share_name, path):
    """List all files in the specified path."""
    try:
        files = []
        file_list = conn.listPath(share_name, path)
        
        for f in file_list:
            if not f.isDirectory and f.filename not in ['.', '..']:
                full_path = path.rstrip('/') + '/' + f.filename
                files.append({
                    'name': f.filename,
                    'path': full_path,
                    'size': f.file_size
                })
        
        print(f"Found {len(files)} files in {path}")
        return files
    except Exception as e:
        print(f"Error listing files: {e}")
        return []

def compute_file_hash(conn, share_name, path, chunk_size=8192):
    """Compute MD5 hash of a file on the SMB share."""
    try:
        md5 = hashlib.md5()
        
        with tempfile.NamedTemporaryFile() as temp_file:
            file_obj = temp_file
            conn.retrieveFile(share_name, path, file_obj)
            file_obj.seek(0)
            
            while True:
                chunk = file_obj.read(chunk_size)
                if not chunk:
                    break
                md5.update(chunk)
        
        return md5.hexdigest()
    except Exception as e:
        print(f"Error computing hash for {path}: {e}")
        return None

def find_duplicate_groups(files):
    """
    Group files that might be duplicates based on naming pattern.
    E.g., "file.txt" and "file 2.txt" would be in the same group.
    """
    # Pattern to match numbered duplicates like " 2", " 3", etc.
    duplicate_pattern = re.compile(r'^(.+?)( \d+)?(\.[^.]+)$')
    
    groups = defaultdict(list)
    
    for file in files:
        match = duplicate_pattern.match(file['name'])
        if match:
            base_name = match.group(1)
            extension = match.group(3) if match.group(3) else ''
            number = match.group(2)
            
            # Create a group key from base name and extension
            group_key = f"{base_name}{extension}"
            groups[group_key].append({
                'file': file,
                'number': number if number else '',
                'sort_key': 0 if not number else int(number.strip())
            })
    
    # Filter to only groups with multiple files
    duplicate_groups = {k: v for k, v in groups.items() if len(v) > 1}
    
    return duplicate_groups

def verify_and_remove_duplicates(conn, share_name, duplicate_groups, dry_run=True):
    """
    Verify files are identical and remove duplicates.
    Keeps the file without a number (the "original").
    """
    results = {
        'verified_duplicates': [],
        'different_files': [],
        'errors': []
    }
    
    for group_name, files in duplicate_groups.items():
        print(f"\n{'='*80}")
        print(f"Checking group: {group_name}")
        print(f"Files in group: {len(files)}")
        
        # Sort by number (original file first)
        files_sorted = sorted(files, key=lambda x: x['sort_key'])
        
        # Display all files in group
        for f in files_sorted:
            print(f"  - {f['file']['name']} (Size: {f['file']['size']:,} bytes)")
        
        # Compare all files by size first
        sizes = [f['file']['size'] for f in files_sorted]
        if len(set(sizes)) > 1:
            print("  ⚠️  FILES HAVE DIFFERENT SIZES - NOT DUPLICATES")
            results['different_files'].append(group_name)
            continue
        
        print(f"  ✓ All files have same size: {sizes[0]:,} bytes")
        
        # Compute hashes for all files
        print("  Computing file hashes...")
        file_hashes = []
        for f in files_sorted:
            hash_val = compute_file_hash(conn, share_name, f['file']['path'])
            if hash_val:
                file_hashes.append(hash_val)
                print(f"    {f['file']['name']}: {hash_val}")
            else:
                print(f"    ⚠️  Failed to hash {f['file']['name']}")
                results['errors'].append(f['file']['name'])
        
        # Check if all hashes match
        if len(file_hashes) != len(files_sorted):
            print("  ⚠️  Could not hash all files - skipping")
            continue
        
        if len(set(file_hashes)) > 1:
            print("  ⚠️  FILES HAVE DIFFERENT CONTENT - NOT DUPLICATES")
            results['different_files'].append(group_name)
            continue
        
        print("  ✓ All files have identical content!")
        
        # Keep the first file (original without number), delete the rest
        original = files_sorted[0]
        duplicates = files_sorted[1:]
        
        print(f"  📌 Keeping: {original['file']['name']}")
        
        for dup in duplicates:
            dup_name = dup['file']['name']
            dup_path = dup['file']['path']
            
            if dry_run:
                print(f"  🗑️  Would delete: {dup_name}")
            else:
                try:
                    conn.deleteFiles(share_name, dup_path)
                    print(f"  ✅ Deleted: {dup_name}")
                    results['verified_duplicates'].append(dup_name)
                except Exception as e:
                    print(f"  ❌ Error deleting {dup_name}: {e}")
                    results['errors'].append(dup_name)
    
    return results

def main():
    print("SMB Duplicate File Finder and Remover")
    print("=" * 80)
    
    # Connect to SMB
    conn = connect_to_smb()
    if not conn:
        print("Failed to connect to SMB share")
        return
    
    try:
        # List all files
        files = list_files(conn, SMB_SHARE, SMB_PATH)
        
        if not files:
            print("No files found or error accessing directory")
            return
        
        # Find potential duplicate groups
        print("\nAnalyzing files for duplicates...")
        duplicate_groups = find_duplicate_groups(files)
        
        if not duplicate_groups:
            print("\n✅ No duplicate files found!")
            return
        
        print(f"\n🔍 Found {len(duplicate_groups)} potential duplicate groups")
        
        # First pass: DRY RUN
        print("\n" + "="*80)
        print("DRY RUN - No files will be deleted")
        print("="*80)
        
        results = verify_and_remove_duplicates(conn, SMB_SHARE, duplicate_groups, dry_run=True)
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total duplicate groups found: {len(duplicate_groups)}")
        print(f"Different files (not duplicates): {len(results['different_files'])}")
        print(f"Errors: {len(results['errors'])}")
        
        # Ask for confirmation to actually delete
        print("\n" + "="*80)
        response = input("\nDo you want to DELETE the duplicate files? (yes/no): ").strip().lower()
        
        if response == 'yes':
            print("\n🗑️  DELETING DUPLICATES...")
            results = verify_and_remove_duplicates(conn, SMB_SHARE, duplicate_groups, dry_run=False)
            print(f"\n✅ Deleted {len(results['verified_duplicates'])} duplicate files")
        else:
            print("\n❌ Deletion cancelled. No files were deleted.")
    
    finally:
        conn.close()
        print("\nConnection closed.")

if __name__ == "__main__":
    main()
