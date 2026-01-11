#!/usr/bin/env python3
"""
Secure Key File Encryption - HIPAA Compliant
Encrypts the de-identification key CSV with AES-256 encryption

This script:
1. Takes your deid_key.csv file
2. Encrypts it with AES-256 (HIPAA-compliant standard)
3. Creates password-protected zip file
4. Optionally securely deletes the original
5. Creates backup instructions

CRITICAL: Store the password in a secure location!
Without the password, the key cannot be recovered.

Usage:
    python3 encrypt_key_file.py
    python3 encrypt_key_file.py --key_file /path/to/your_key.csv
    python3 encrypt_key_file.py --key_file /path/to/key.csv --output_dir /path/to/output
"""

import os
import sys
import getpass
import shutil
import argparse
from pathlib import Path
from datetime import datetime

try:
    import pyzipper
    PYZIPPER_AVAILABLE = True
except ImportError:
    PYZIPPER_AVAILABLE = False

# ============================================================================
# FUNCTIONS
# ============================================================================

def verify_key_file(filepath):
    """Verify the key file exists and has required columns"""
    if not os.path.exists(filepath):
        print(f"❌ Error: Key file not found at {filepath}")
        return False
    
    try:
        import pandas as pd
        df = pd.read_csv(filepath)
        required_cols = ['mrn', 'UID', 'date_shift_days']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            print(f"⚠️  Warning: Key file missing columns: {missing}")
            return False
        
        print(f"✓ Key file verified: {len(df)} patients")
        return True
        
    except Exception as e:
        print(f"❌ Error reading key file: {e}")
        return False

def get_strong_password():
    """Prompt for and verify a strong password"""
    print("\n" + "="*70)
    print("PASSWORD REQUIREMENTS (HIPAA Compliance)")
    print("="*70)
    print("• Minimum 12 characters")
    print("• Mix of uppercase, lowercase, numbers, and symbols")
    print("• This password is CRITICAL - without it, key cannot be recovered")
    print("• Store in a secure password manager!")
    print("="*70 + "\n")
    
    while True:
        password = getpass.getpass("Enter encryption password: ")
        password_confirm = getpass.getpass("Confirm password: ")
        
        if password != password_confirm:
            print("❌ Passwords don't match. Try again.\n")
            continue
        
        if len(password) < 12:
            print("❌ Password must be at least 12 characters.\n")
            continue
        
        # Check complexity
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_symbol = any(not c.isalnum() for c in password)
        
        if not (has_upper and has_lower and has_digit and has_symbol):
            print("❌ Password must contain uppercase, lowercase, numbers, and symbols.\n")
            continue
        
        print("✓ Strong password accepted\n")
        return password

def encrypt_with_pyzipper(source_file, output_file, password):
    """Encrypt file using pyzipper with AES-256"""
    try:
        with pyzipper.AESZipFile(
            output_file,
            'w',
            compression=pyzipper.ZIP_DEFLATED,
            encryption=pyzipper.WZ_AES
        ) as zf:
            zf.setpassword(password.encode('utf-8'))
            zf.write(source_file, os.path.basename(source_file))
        
        return True
    except Exception as e:
        print(f"❌ Encryption error: {e}")
        return False

def verify_encryption(zip_file, password, original_file):
    """Verify the encrypted file can be decrypted"""
    try:
        with pyzipper.AESZipFile(zip_file) as zf:
            zf.setpassword(password.encode('utf-8'))
            names = zf.namelist()
            if os.path.basename(original_file) in names:
                zf.read(os.path.basename(original_file))
                return True
        return False
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def secure_delete(filepath):
    """Securely delete a file (overwrite then delete)"""
    try:
        file_size = os.path.getsize(filepath)
        with open(filepath, 'wb') as f:
            f.write(os.urandom(file_size))
        os.remove(filepath)
        return True
    except Exception as e:
        print(f"❌ Secure deletion failed: {e}")
        return False

def create_backup_instructions(output_dir, encrypted_file):
    """Create instructions document"""
    instructions_file = os.path.join(output_dir, "KEY_FILE_INSTRUCTIONS.txt")
    
    content = f"""
================================================================================
ENCRYPTED DE-IDENTIFICATION KEY - CRITICAL INFORMATION
================================================================================

Encrypted File: {os.path.basename(encrypted_file)}
Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Location: {encrypted_file}

================================================================================
⚠️  CRITICAL SECURITY WARNINGS
================================================================================

1. This file contains the ONLY mapping between real patient identifiers and
   de-identified study codes.

2. WITHOUT this file AND the password:
   • De-identified data CANNOT be re-identified
   • Cannot link back to medical records
   • Cannot correct errors in de-identification
   • Data is permanently anonymized

3. PROTECT BOTH:
   • The encrypted file (store securely)
   • The password (store separately in password manager)

================================================================================
PASSWORD STORAGE - REQUIRED ACTIONS
================================================================================

✓ IMMEDIATELY store password in secure location:
  
  Option 1: Password Manager (RECOMMENDED)
    • 1Password, LastPass, Bitwarden, Dashlane, etc.
    • Entry name: "PHI De-identification Key - [Study Name]"
    • Add file location and date in notes
  
  Option 2: Institutional Key Management
    • Follow your institution's policies
    • Register as "Research PHI Encryption Key"
    • Document project and IRB number
  
  Option 3: Physical Backup (ADDITIONAL only)
    • Write password on paper
    • Store in locked safe
    • NEVER store with encrypted file

================================================================================
BACKUP STRATEGY - IMPLEMENT IMMEDIATELY
================================================================================

Create AT LEAST 3 copies of the encrypted file:

1. PRIMARY: {encrypted_file}
   • Keep on local machine
   • For regular use

2. BACKUP #1: Institutional Secure Storage
   • Research data repository
   • Institutional file server
   • Access-controlled location

3. BACKUP #2: Offsite Encrypted Drive
   • External encrypted hard drive
   • Store at different physical location
   • Test quarterly

4. BACKUP #3: (Optional but recommended)
   • Institutional backup system
   • Cloud storage with institutional encryption
   • Follow data governance policies

⚠️  NEVER store password with encrypted file
⚠️  Test decryption quarterly to verify everything works

================================================================================
HOW TO DECRYPT WHEN NEEDED
================================================================================

Method 1: Using companion script
    python3 decrypt_key.py
    # Enter password when prompted

Method 2: Using command line (if pyzipper installed)
    python3 -c "import pyzipper; z=pyzipper.AESZipFile('{os.path.basename(encrypted_file)}'); 
    z.setpassword(input('Password: ').encode()); z.extractall()"

Method 3: Using 7-Zip (if installed)
    7z x {os.path.basename(encrypted_file)}
    # Enter password when prompted

Method 4: macOS (may not support AES-256)
    # Double-click file, enter password
    # NOTE: May not work with AES-256, use Method 1 or 2

================================================================================
QUARTERLY VERIFICATION CHECKLIST
================================================================================

Every 3 months, verify:

[ ] Encrypted file exists in all backup locations
[ ] Can access password from password manager
[ ] Test decryption (decrypt to temp location, then delete)
[ ] Verify file is not corrupted
[ ] Update backup locations if needed
[ ] Document verification date

================================================================================
EMERGENCY PROCEDURES
================================================================================

IF PASSWORD IS LOST:
  • There is NO recovery method
  • Data becomes permanently anonymized
  • Contact IRB immediately if re-identification needed

IF FILE IS LOST:
  • Check all backup locations
  • Contact IT for backup restoration
  • If all copies lost, data is permanently anonymized

IF SECURITY BREACH SUSPECTED:
  • Immediately change password (create new encrypted file)
  • Document incident per institutional policy
  • Contact IRB and data security officer
  • Assess if data compromise occurred

================================================================================
RETENTION POLICY
================================================================================

• Keep encrypted key for duration of study + retention period
• Follow institutional data retention policies
• Typically: Study duration + 7 years minimum
• Check IRB requirements for your specific study

When study is complete and retention period expires:
1. Securely delete all copies of encrypted file
2. Delete password from password manager
3. Document destruction per institutional policy

================================================================================
COMPLIANCE NOTES
================================================================================

This encryption meets HIPAA requirements:
• AES-256 encryption (NIST approved)
• Password-protected
• Secure key management procedures documented
• Backup and recovery procedures in place

Maintain this documentation with:
• IRB records
• Data management plan
• Study regulatory binder

================================================================================
CONTACT INFORMATION
================================================================================

For questions about:
• Data security: Contact your institutional IT security office
• IRB compliance: Contact your IRB administrator
• Key recovery: NO RECOVERY POSSIBLE - prevention is critical

================================================================================
VERSION AND AUDIT TRAIL
================================================================================

File encrypted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Encryption method: AES-256
Tool: pyzipper Python library
Instructions version: 1.0

Maintain log of:
• When file accessed
• Who accessed it
• Purpose of access
• Verification checks performed

================================================================================
"""
    
    with open(instructions_file, 'w') as f:
        f.write(content)
    
    return instructions_file

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Encrypt de-identification key file with AES-256',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default location (prompts if file not found)
  python3 encrypt_key_file.py
  
  # Specify key file path
  python3 encrypt_key_file.py --key_file /path/to/deid_key.csv
  
  # Specify both key file and output directory
  python3 encrypt_key_file.py --key_file /path/to/key.csv --output_dir /path/to/output
        """
    )
    
    parser.add_argument(
        '--key_file',
        type=str,
        help='Path to de-identification key CSV file (default: ./deid_key.csv or prompt)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Directory to save encrypted file (default: same as key file location)'
    )
    
    parser.add_argument(
        '--output_name',
        type=str,
        default='deid_key_ENCRYPTED.zip',
        help='Name for encrypted file (default: deid_key_ENCRYPTED.zip)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("SECURE KEY FILE ENCRYPTION - HIPAA COMPLIANT")
    print("="*70 + "\n")
    
    # Check if pyzipper is available
    if not PYZIPPER_AVAILABLE:
        print("❌ pyzipper library not installed")
        print("\nTo install:")
        print("    pip install pyzipper --break-system-packages")
        print("\nOR use manual encryption with 7-Zip (instructions below)")
        print_manual_instructions()
        return
    
    # Determine key file path
    if args.key_file:
        key_file = args.key_file
    else:
        # Try common locations
        common_locations = [
            './deid_key.csv',
            '/path/to/user/Desktop/deid_key.csv',
            os.path.expanduser('~/Desktop/deid_key.csv')
        ]
        
        key_file = None
        for location in common_locations:
            if os.path.exists(location):
                key_file = location
                print(f"Found key file at: {key_file}")
                break
        
        if not key_file:
            print("Key file not found in common locations.")
            key_file = input("Enter path to your key file: ").strip()
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Use same directory as key file
        output_dir = os.path.dirname(os.path.abspath(key_file))
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nKey file: {key_file}")
    print(f"Output directory: {output_dir}\n")
    
    # Verify key file exists
    print("[1/6] Verifying key file...")
    if not verify_key_file(key_file):
        return
    
    # Get encryption password
    print("\n[2/6] Setting encryption password...")
    password = get_strong_password()
    
    # Create output path
    output_file = os.path.join(output_dir, args.output_name)
    
    # Encrypt the file
    print("[3/6] Encrypting with AES-256...")
    if not encrypt_with_pyzipper(key_file, output_file, password):
        print("❌ Encryption failed!")
        return
    
    print(f"✓ Encrypted file created: {output_file}")
    
    # Verify encryption
    print("\n[4/6] Verifying encryption...")
    if not verify_encryption(output_file, password, key_file):
        print("❌ Verification failed! Encryption may be corrupted.")
        return
    
    print("✓ Encryption verified successfully")
    
    # Create instructions
    print("\n[5/6] Creating instructions document...")
    instructions_file = create_backup_instructions(output_dir, output_file)
    print(f"✓ Instructions saved: {instructions_file}")
    
    # Ask about deleting original
    print("\n[6/6] Original file handling...")
    print("⚠️  IMPORTANT: Should the original unencrypted file be deleted?")
    print("\nOptions:")
    print("  1. DELETE original (secure - recommended after creating backups)")
    print("  2. KEEP original (for now - you can delete later)")
    
    while True:
        choice = input("\nEnter choice (1 or 2): ").strip()
        if choice == '1':
            print("\n⚠️  FINAL WARNING: About to securely delete original file")
            print("    Ensure you have:")
            print("    • Tested the encrypted file works")
            print("    • Saved the password securely")
            print("    • Created backup copies")
            confirm = input("\nType 'DELETE' to confirm: ").strip()
            
            if confirm == 'DELETE':
                print("\nSecurely deleting original file...")
                if secure_delete(key_file):
                    print("✓ Original file securely deleted")
                else:
                    print("⚠️  Could not securely delete. Manually delete when ready.")
            else:
                print("Deletion cancelled. Original file retained.")
            break
        elif choice == '2':
            print("\n✓ Original file retained.")
            print("  Remember to delete it after creating backups!")
            break
        else:
            print("Invalid choice. Enter 1 or 2.")
    
    # Final summary
    print("\n" + "="*70)
    print("ENCRYPTION COMPLETE")
    print("="*70)
    print(f"\n✓ Encrypted file: {output_file}")
    print(f"✓ Instructions:   {instructions_file}")
    print("\nNEXT STEPS:")
    print("1. IMMEDIATELY store password in secure location")
    print("2. Test decryption to verify password works")
    print("3. Create backup copies of encrypted file")
    print("4. Read the instructions document carefully")
    print("5. Delete original unencrypted file (if not done)")
    print("\n⚠️  Store password and encrypted file in SEPARATE secure locations!")
    print("="*70 + "\n")

def print_manual_instructions():
    """Print manual encryption instructions if pyzipper not available"""
    print("\n" + "="*70)
    print("MANUAL ENCRYPTION INSTRUCTIONS (7-Zip)")
    print("="*70)
    print("\nInstall 7-Zip:")
    print("    brew install p7zip")
    print("\nEncrypt your key file:")
    print("    7z a -p -mem=AES256 deid_key_ENCRYPTED.zip /path/to/your_key_file.csv")
    print("\n• You will be prompted for password")
    print("• Use a strong password (12+ chars, mixed case, numbers, symbols)")
    print("• Store password in password manager")
    print("\nTo decrypt later:")
    print("    7z x deid_key_ENCRYPTED.zip")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
