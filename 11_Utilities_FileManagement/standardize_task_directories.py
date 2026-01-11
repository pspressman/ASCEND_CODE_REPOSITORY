#!/usr/bin/env python3
"""
Directory Name Standardizer for ASCEND Cleaned Audio
Renames task directories to standardized format for simplified filtering

Usage:
    python standardize_task_directories.py --root /path/to/CleanedSplicedAudio --dry-run
    python standardize_task_directories.py --root /path/to/CleanedSplicedAudio --manifest log.csv
"""

import os
import sys
import argparse
import csv
from pathlib import Path
from datetime import datetime


# Standardization rules for task names
TASK_STANDARDIZATION = {
    # Standard naming (no changes needed)
    'ConflictConv': 'ConflictConv',
    'GrandfatherPassage': 'GrandfatherPassage',
    'MotorSpeechEval': 'MotorSpeechEval',
    'PicnicDescription': 'PicnicDescription',
    'SpontSpeech': 'SpontSpeech',
    
    # Names with spaces (remove spaces, standardize)
    'Grandfather Passage': 'GrandfatherPassage',
    'Motor Speech Evaluation': 'MotorSpeechEval',
    'Picnic Description': 'PicnicDescription',
    'Spontaneous Speech': 'SpontSpeech',
    'Conflict Conversation': 'ConflictConv',
    
    # CSA-Research special cases - keep Mac/PC and Coordinator variants separate
    'Mac_Audacity_10MinConvo_withCoordinatorSpeech': 'ConflictConv_Mac_WithCoord',
    'Mac_Audacity_10MinuteConvo_withoutCoordinatorSpeech': 'ConflictConv_Mac_NoCoord',
    'PC_10MinuteConversations_WithoutCoordinatorSpeech': 'ConflictConv_PC_NoCoord',
    
    'Mac_Audacity_Grandfather_Passage': 'GrandfatherPassage_Mac',
    'PC_Audacity_Grandfather_Passage': 'GrandfatherPassage_PC',
    
    'Mac_Audacity_Motor_Speech_Evaluation': 'MotorSpeechEval_Mac',
    'PC_Audacity_Motor_Speech_Evaluation': 'MotorSpeechEval_PC',
    
    'Mac_Audacity_Picnic_Description': 'PicnicDescription_Mac',
    'PC_Audacity_Picnic_Description': 'PicnicDescription_PC',
    
    'Mac_Audacity_Spontaneous_Speech': 'SpontSpeech_Mac',
    'PC_Audacity_Spontaneous_Speech': 'SpontSpeech_PC',
}


def find_task_directories(root_path):
    """
    Recursively find all task directories that need standardization.
    Returns list of (full_path, dir_name, parent_path, cohort_name) tuples.
    """
    root = Path(root_path)
    task_dirs = []
    
    print(f"\n{'='*80}")
    print("SCANNING FOR TASK DIRECTORIES")
    print(f"{'='*80}\n")
    
    # Walk through directory structure
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip forPraat directory
        if 'forPraat' in dirpath:
            continue
            
        current_path = Path(dirpath)
        
        # Check if any subdirectory names match our standardization rules
        for dirname in dirnames:
            if dirname in TASK_STANDARDIZATION:
                full_path = current_path / dirname
                
                # Determine cohort name (top-level directory under root)
                try:
                    cohort = current_path.relative_to(root).parts[0]
                except (ValueError, IndexError):
                    cohort = "Unknown"
                
                task_dirs.append({
                    'full_path': full_path,
                    'original_name': dirname,
                    'parent_path': current_path,
                    'cohort': cohort,
                    'standardized_name': TASK_STANDARDIZATION[dirname]
                })
    
    return task_dirs


def preview_changes(task_dirs):
    """Display what would be renamed without making changes."""
    print(f"\n{'='*80}")
    print("PREVIEW: DIRECTORIES TO BE RENAMED")
    print(f"{'='*80}\n")
    
    changes_needed = []
    no_change_needed = []
    
    for task in task_dirs:
        if task['original_name'] != task['standardized_name']:
            changes_needed.append(task)
        else:
            no_change_needed.append(task)
    
    if changes_needed:
        print(f"📝 {len(changes_needed)} directories need renaming:\n")
        for task in changes_needed:
            print(f"  Cohort: {task['cohort']}")
            print(f"    FROM: {task['original_name']}")
            print(f"    TO:   {task['standardized_name']}")
            print(f"    Path: {task['full_path']}")
            print()
    else:
        print("✅ No directories need renaming - all already standardized!\n")
    
    if no_change_needed:
        print(f"✅ {len(no_change_needed)} directories already standardized:\n")
        for task in no_change_needed:
            print(f"  {task['cohort']}: {task['original_name']}")
    
    return changes_needed


def rename_directories(task_dirs, dry_run=False):
    """Rename directories according to standardization rules."""
    
    changes_needed = [t for t in task_dirs if t['original_name'] != t['standardized_name']]
    
    if not changes_needed:
        print("\n✅ All directories already standardized. Nothing to do!")
        return []
    
    if not dry_run:
        print(f"\n{'='*80}")
        print("RENAMING DIRECTORIES")
        print(f"{'='*80}\n")
        
        response = input(f"About to rename {len(changes_needed)} directories. Continue? [y/N]: ")
        if response.lower() != 'y':
            print("❌ Aborted by user.")
            return []
    
    renamed = []
    errors = []
    
    for i, task in enumerate(changes_needed, 1):
        old_path = task['full_path']
        new_path = task['parent_path'] / task['standardized_name']
        
        print(f"[{i}/{len(changes_needed)}] {task['cohort']}: {task['original_name']} → {task['standardized_name']}")
        
        if dry_run:
            print(f"  [DRY RUN] Would rename: {old_path}")
            print(f"  [DRY RUN]          to: {new_path}\n")
            renamed.append({
                'cohort': task['cohort'],
                'original_name': task['original_name'],
                'standardized_name': task['standardized_name'],
                'old_path': str(old_path),
                'new_path': str(new_path),
                'status': 'dry_run'
            })
        else:
            try:
                # Check if target already exists
                if new_path.exists():
                    print(f"  ⚠️  WARNING: Target already exists: {new_path}")
                    print(f"  ❌ SKIPPING to avoid overwrite\n")
                    errors.append({
                        'cohort': task['cohort'],
                        'original_name': task['original_name'],
                        'standardized_name': task['standardized_name'],
                        'old_path': str(old_path),
                        'new_path': str(new_path),
                        'status': 'error',
                        'error': 'Target already exists'
                    })
                    continue
                
                # Rename the directory
                old_path.rename(new_path)
                print(f"  ✅ Renamed successfully\n")
                
                renamed.append({
                    'cohort': task['cohort'],
                    'original_name': task['original_name'],
                    'standardized_name': task['standardized_name'],
                    'old_path': str(old_path),
                    'new_path': str(new_path),
                    'status': 'success'
                })
                
            except Exception as e:
                print(f"  ❌ ERROR: {e}\n")
                errors.append({
                    'cohort': task['cohort'],
                    'original_name': task['original_name'],
                    'standardized_name': task['standardized_name'],
                    'old_path': str(old_path),
                    'new_path': str(new_path),
                    'status': 'error',
                    'error': str(e)
                })
    
    return renamed, errors


def save_manifest(renamed, errors, manifest_path):
    """Save renaming manifest to CSV."""
    
    if not renamed and not errors:
        print("\n⚠️  No changes to record in manifest.")
        return
    
    manifest_file = Path(manifest_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    
    all_records = renamed + errors
    
    with open(manifest_file, 'w', newline='') as f:
        fieldnames = ['cohort', 'original_name', 'standardized_name', 'old_path', 'new_path', 'status', 'error']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for record in all_records:
            if 'error' not in record:
                record['error'] = ''
            writer.writerow(record)
    
    print(f"\n📄 Manifest saved: {manifest_file}")
    print(f"   Total records: {len(all_records)}")
    print(f"   Successful: {len(renamed)}")
    if errors:
        print(f"   Errors: {len(errors)}")


def main():
    parser = argparse.ArgumentParser(
        description='Standardize task directory names in ASCEND cleaned audio',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview changes without making any modifications
  python standardize_task_directories.py --root /path/to/CleanedSplicedAudio --dry-run
  
  # Actually rename directories and save manifest
  python standardize_task_directories.py --root /path/to/CleanedSplicedAudio --manifest standardization_log.csv
  
  # Preview with custom manifest location
  python standardize_task_directories.py --root /path/to/CleanedSplicedAudio --dry-run --manifest preview.csv
        """
    )
    
    parser.add_argument('--root', required=True, 
                       help='Root directory containing CleanedSplicedAudio')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview changes without actually renaming')
    parser.add_argument('--manifest', default=None,
                       help='Path to save manifest CSV (default: standardization_manifest_TIMESTAMP.csv)')
    
    args = parser.parse_args()
    
    # Validate root directory exists
    root_path = Path(args.root)
    if not root_path.exists():
        print(f"❌ ERROR: Root directory does not exist: {root_path}")
        sys.exit(1)
    
    # Generate default manifest filename if not provided
    if args.manifest is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.manifest = f"standardization_manifest_{timestamp}.csv"
    
    print(f"\n{'='*80}")
    print("ASCEND DIRECTORY NAME STANDARDIZER")
    print(f"{'='*80}")
    print(f"Root directory: {root_path}")
    print(f"Mode: {'DRY RUN (preview only)' if args.dry_run else 'LIVE (will rename)'}")
    print(f"Manifest: {args.manifest}")
    print(f"{'='*80}")
    
    # Find all task directories
    task_dirs = find_task_directories(root_path)
    
    if not task_dirs:
        print("\n⚠️  No task directories found matching standardization rules.")
        print("   Either directory structure is different than expected, or all already standardized.")
        sys.exit(0)
    
    print(f"\n✅ Found {len(task_dirs)} task directories")
    
    # Preview what would change
    changes_needed = preview_changes(task_dirs)
    
    if not changes_needed:
        print("\n🎉 All directories already have standardized names!")
        sys.exit(0)
    
    # Rename directories (or simulate if dry-run)
    if args.dry_run:
        renamed, errors = rename_directories(task_dirs, dry_run=True)
    else:
        renamed, errors = rename_directories(task_dirs, dry_run=False)
    
    # Save manifest
    save_manifest(renamed, errors, args.manifest)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    if args.dry_run:
        print(f"✅ Dry run complete - no actual changes made")
        print(f"   {len(renamed)} directories would be renamed")
    else:
        print(f"✅ Standardization complete")
        print(f"   {len(renamed)} directories renamed successfully")
        if errors:
            print(f"   ⚠️  {len(errors)} errors occurred (see manifest)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
