#!/usr/bin/env python3
"""
Decrypt De-identification Key File
Safely decrypts the AES-256 encrypted key file when needed

Usage:
    python3 decrypt_key.py
    python3 decrypt_key.py --encrypted /path/to/encrypted.zip
    python3 decrypt_key.py --encrypted /path/to/encrypted.zip --output_dir /path/to/output
"""

import os
import sys
import getpass
import argparse
from pathlib import Path

try:
    import pyzipper
    PYZIPPER_AVAILABLE = True
except ImportError:
    PYZIPPER_AVAILABLE = False

# ============================================================================
# FUNCTIONS
# ============================================================================

def decrypt_file(encrypted_file, output_dir, password):
    """Decrypt the encrypted key file"""
    try:
        with pyzipper.AESZipFile(encrypted_file) as zf:
            zf.setpassword(password.encode('utf-8'))
            
            # Get list of files in archive
            file_list = zf.namelist()
            
            if not file_list:
                print("❌ No files found in encrypted archive")
                return False
            
            # Extract all files
            print(f"\nExtracting {len(file_list)} file(s)...")
            for filename in file_list:
                print(f"  • {filename}")
                zf.extract(filename, output_dir)
            
            return True
            
    except RuntimeError as e:
        if "Bad password" in str(e):
            print("❌ Incorrect password")
        else:
            print(f"❌ Decryption error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def verify_decrypted_file(filepath):
    """Verify the decrypted key file is valid"""
    try:
        import pandas as pd
        df = pd.read_csv(filepath)
        required_cols = ['mrn', 'UID', 'date_shift_days']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            print(f"⚠️  Warning: Decrypted file missing columns: {missing}")
            return False
        
        print(f"✓ Decrypted key file verified: {len(df)} patients")
        return True
        
    except Exception as e:
        print(f"⚠️  Could not verify decrypted file: {e}")
        return False

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Decrypt the de-identification key file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect encrypted file in common locations
  python3 decrypt_key.py
  
  # Specify encrypted file path
  python3 decrypt_key.py --encrypted /path/to/encrypted.zip
  
  # Specify both encrypted file and output directory
  python3 decrypt_key.py --encrypted /path/to/encrypted.zip --output_dir /path/to/output
        """
    )
    
    parser.add_argument(
        '--encrypted',
        type=str,
        help='Path to encrypted file (default: auto-detect in common locations)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Output directory for decrypted file (default: same as encrypted file location)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("DE-IDENTIFICATION KEY FILE DECRYPTION")
    print("="*70 + "\n")
    
    # Check if pyzipper is available
    if not PYZIPPER_AVAILABLE:
        print("❌ pyzipper library not installed")
        print("\nTo install:")
        print("    pip install pyzipper --break-system-packages")
        print("\nOR use 7-Zip:")
        if args.encrypted:
            print(f"    7z x {args.encrypted}")
        else:
            print(f"    7z x deid_key_ENCRYPTED.zip")
        return 1
    
    # Determine encrypted file path
    if args.encrypted:
        encrypted_file = args.encrypted
    else:
        # Try common locations
        common_locations = [
            './deid_key_ENCRYPTED.zip',
            '~/Desktop/deid_key_ENCRYPTED.zip',
            '/path/to/user/Desktop/deid_key_ENCRYPTED.zip',
            os.path.expanduser('~/Desktop/deid_key_ENCRYPTED.zip')
        ]
        
        encrypted_file = None
        for location in common_locations:
            expanded_location = os.path.expanduser(location)
            if os.path.exists(expanded_location):
                encrypted_file = expanded_location
                print(f"Found encrypted file at: {encrypted_file}")
                break
        
        if not encrypted_file:
            print("Encrypted file not found in common locations.")
            encrypted_file = input("Enter path to encrypted file: ").strip()
    
    # Check if encrypted file exists
    if not os.path.exists(encrypted_file):
        print(f"❌ Encrypted file not found: {encrypted_file}")
        return 1
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Use same directory as encrypted file
        output_dir = os.path.dirname(os.path.abspath(encrypted_file))
    
    print(f"Encrypted file: {encrypted_file}")
    print(f"Output directory: {output_dir}\n")
    
    # Get password
    password = getpass.getpass("Enter decryption password: ")
    
    # Decrypt
    print("\nDecrypting...")
    if decrypt_file(encrypted_file, output_dir, password):
        print("\n✓ Decryption successful")
        
        # Try to verify the decrypted file
        potential_csv = os.path.join(output_dir, "deid_key.csv")
        if os.path.exists(potential_csv):
            print("\nVerifying decrypted file...")
            verify_decrypted_file(potential_csv)
        
        print("\n" + "="*70)
        print("⚠️  SECURITY REMINDERS")
        print("="*70)
        print("• Decrypted file now contains UNENCRYPTED PHI")
        print("• Use it only as needed")
        print("• DELETE when finished (or re-encrypt)")
        print("• Do NOT leave unencrypted on shared computers")
        print("• Do NOT email or transmit unencrypted")
        print("="*70 + "\n")
        
        return 0
    else:
        print("\n❌ Decryption failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
