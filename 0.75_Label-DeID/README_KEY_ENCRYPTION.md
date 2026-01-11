# De-identification Key File Security - HIPAA Compliant Encryption

## Why This Matters

Your `deid_key.csv` file is **THE MOST CRITICAL FILE** in your entire de-identification system:

- Contains **MRN → UID mappings** (the only way to re-identify data)
- Contains **date shift values** (the only way to recover real dates)
- Without this file: De-identified data becomes **permanently anonymous**
- If compromised: **PHI breach**, HIPAA violation, IRB issues

**Bottom line:** This file MUST be encrypted with HIPAA-compliant encryption.

---

## What These Scripts Do

### `encrypt_key_file.py` - Encrypt Your Key File

**Features:**
- ✅ **AES-256 encryption** (HIPAA-compliant standard)
- ✅ **Strong password enforcement** (12+ chars, mixed complexity)
- ✅ **Verification step** (ensures decryption works)
- ✅ **Secure deletion option** (overwrites original before deleting)
- ✅ **Comprehensive instructions** (auto-generated backup guide)

**Creates:**
1. `deid_key_ENCRYPTED.zip` - Your encrypted key file
2. `KEY_FILE_INSTRUCTIONS.txt` - Critical security and backup instructions

### `decrypt_key.py` - Decrypt When Needed

**Features:**
- ✅ Safe decryption with password verification
- ✅ Validates decrypted file structure
- ✅ Security reminders after decryption

---

## Quick Start

### Step 1: Install Required Library

```bash
pip install pyzipper --break-system-packages
```

**What is pyzipper?**
- Python library for AES-256 encryption
- HIPAA-compliant encryption standard
- Industry standard for PHI protection

### Step 2: Encrypt Your Key File

```bash
python3 encrypt_key_file.py
```

**You'll be prompted for:**
1. Password verification (must be strong)
2. Whether to delete original

**Example session:**
```
======================================================================
SECURE KEY FILE ENCRYPTION - HIPAA COMPLIANT
======================================================================

[1/6] Verifying key file...
✓ Key file verified: 150 patients

[2/6] Setting encryption password...

======================================================================
PASSWORD REQUIREMENTS (HIPAA Compliance)
======================================================================
• Minimum 12 characters
• Mix of uppercase, lowercase, numbers, and symbols
• This password is CRITICAL - without it, key cannot be recovered
• Store in a secure password manager!
======================================================================

Enter encryption password: ****************
Confirm password: ****************
✓ Strong password accepted

[3/6] Encrypting with AES-256...
✓ Encrypted file created: /Users/peterpressman/Desktop/deid_key_ENCRYPTED.zip

[4/6] Verifying encryption...
✓ Encryption verified successfully

[5/6] Creating instructions document...
✓ Instructions saved: /Users/peterpressman/Desktop/KEY_FILE_INSTRUCTIONS.txt

[6/6] Original file handling...
⚠️  IMPORTANT: Should the original unencrypted file be deleted?

Options:
  1. DELETE original (secure - recommended after creating backups)
  2. KEEP original (for now - you can delete later)

Enter choice (1 or 2): 2

✓ Original file retained.
  Remember to delete it after creating backups!

======================================================================
ENCRYPTION COMPLETE
======================================================================

✓ Encrypted file: /Users/peterpressman/Desktop/deid_key_ENCRYPTED.zip
✓ Instructions:   /Users/peterpressman/Desktop/KEY_FILE_INSTRUCTIONS.txt

NEXT STEPS:
1. IMMEDIATELY store password in secure location
2. Test decryption to verify password works
3. Create backup copies of encrypted file
4. Read the instructions document carefully
5. Delete original unencrypted file (if not done)

⚠️  Store password and encrypted file in SEPARATE secure locations!
======================================================================
```

### Step 3: Store Password Securely

**IMMEDIATELY after encrypting, store password in ONE of these:**

#### Option 1: Password Manager (RECOMMENDED)
- **1Password**, LastPass, Bitwarden, Dashlane, etc.
- Create entry: "PHI De-identification Key - [Your Study]"
- Add notes: File location, creation date, study name

#### Option 2: Institutional Key Management
- Follow your institution's policies for research keys
- Register as "PHI Encryption Key"
- Document IRB number and project

#### Option 3: Physical Backup (ADDITIONAL ONLY)
- Write password on paper
- Store in locked safe or safety deposit box
- **NEVER** store with encrypted file

### Step 4: Test Decryption

**CRITICAL:** Test immediately to ensure password works!

```bash
python3 decrypt_key.py
```

**Example:**
```
======================================================================
DE-IDENTIFICATION KEY FILE DECRYPTION
======================================================================

Encrypted file: /Users/peterpressman/Desktop/deid_key_ENCRYPTED.zip
Output directory: /Users/peterpressman/Desktop

Enter decryption password: ****************

Decrypting...

Extracting 1 file(s)...
  • deid_key.csv

✓ Decryption successful

Verifying decrypted file...
✓ Decrypted key file verified: 150 patients

======================================================================
⚠️  SECURITY REMINDERS
======================================================================
• Decrypted file now contains UNENCRYPTED PHI
• Use it only as needed
• DELETE when finished (or re-encrypt)
• Do NOT leave unencrypted on shared computers
• Do NOT email or transmit unencrypted
======================================================================
```

**Then delete the test decrypted file:**
```bash
rm /Users/peterpressman/Desktop/deid_key.csv
```

### Step 5: Create Backups

Create **at least 3 copies** of the encrypted file:

```bash
# Backup 1: External drive
cp deid_key_ENCRYPTED.zip /Volumes/BackupDrive/SecureKeys/

# Backup 2: Institutional storage
cp deid_key_ENCRYPTED.zip /Volumes/Nexus/ResearchKeys/

# Backup 3: Another secure location
cp deid_key_ENCRYPTED.zip ~/Documents/SecureVault/
```

**Store backups in SEPARATE physical locations!**

### Step 6: Delete Original Unencrypted File

**Only after:**
- ✅ Encrypted file created
- ✅ Password stored securely
- ✅ Decryption tested successfully
- ✅ Backup copies created

```bash
# Secure deletion (overwrites then deletes)
rm -P /Users/peterpressman/Desktop/deid_key.csv

# Or on Linux:
shred -u /Users/peterpressman/Desktop/deid_key.csv
```

---

## Usage in Your Workflow

### When Running De-identification Scripts

Your de-identification scripts need the key file. Two options:

#### Option A: Decrypt Temporarily (More Secure)

```bash
# 1. Decrypt key file
python3 decrypt_key.py

# 2. Run de-identification
python3 comprehensive_deid_pipeline.py

# 3. IMMEDIATELY delete decrypted key
rm /Users/peterpressman/Desktop/deid_key.csv
```

#### Option B: Modify Scripts to Use Encrypted Key

Update your de-identification scripts:

```python
import pyzipper
import pandas as pd
import getpass

def load_encrypted_key(encrypted_path):
    """Load key file from encrypted zip"""
    password = getpass.getpass("Enter key file password: ")
    
    with pyzipper.AESZipFile(encrypted_path) as zf:
        zf.setpassword(password.encode('utf-8'))
        with zf.open('deid_key.csv') as f:
            return pd.read_csv(f)

# In your script, replace:
# key_df = pd.read_csv(key_file)

# With:
key_df = load_encrypted_key("/path/to/deid_key_ENCRYPTED.zip")
```

---

## Security Best Practices

### Do's ✅

- ✅ Use strong password (12+ characters, mixed complexity)
- ✅ Store password in password manager
- ✅ Create multiple backup copies of encrypted file
- ✅ Store backups in separate physical locations
- ✅ Test decryption quarterly
- ✅ Delete unencrypted file after encryption
- ✅ Decrypt only when needed
- ✅ Delete decrypted file immediately after use
- ✅ Document encryption in IRB records
- ✅ Follow institutional data governance policies

### Don'ts ❌

- ❌ Use weak or simple password
- ❌ Store password with encrypted file
- ❌ Email encrypted file without separate password transmission
- ❌ Keep unencrypted key file on shared computers
- ❌ Share password via insecure channels (text, email)
- ❌ Leave decrypted key file on system longer than needed
- ❌ Forget to create backups
- ❌ Forget where you stored the password

---

## Quarterly Verification Checklist

**Every 3 months, verify:**

- [ ] Encrypted file exists in all backup locations
- [ ] Can retrieve password from password manager
- [ ] Test decryption successfully completes
- [ ] Decrypted file structure is valid
- [ ] Delete test decryption immediately
- [ ] Update backup locations if changed
- [ ] Document verification date in study logs

---

## Troubleshooting

### "pyzipper library not installed"

**Solution:**
```bash
pip install pyzipper --break-system-packages
```

**Alternative (using 7-Zip):**
```bash
# Install 7-Zip
brew install p7zip

# Encrypt manually
7z a -p -mem=AES256 deid_key_ENCRYPTED.zip deid_key.csv

# Decrypt manually
7z x deid_key_ENCRYPTED.zip
```

### "Incorrect password"

**Causes:**
- Typo in password
- Wrong password retrieved
- Caps Lock was on

**Solution:**
- Try again carefully
- Verify password from password manager
- If truly lost: **NO RECOVERY POSSIBLE**

### "Encrypted file not found"

**Solution:**
- Check file path in script configuration
- Verify drive is mounted (Nexus, external drives)
- Check backup locations
- Verify file wasn't accidentally deleted

### Password Forgotten

**Hard truth:** There is **NO PASSWORD RECOVERY**.

If password is lost:
1. Data becomes permanently anonymized
2. Cannot re-identify participants
3. Cannot recover real dates
4. Contact IRB immediately if re-identification needed
5. This is why backups and password manager are CRITICAL

---

## HIPAA Compliance Notes

This encryption setup meets HIPAA requirements:

### Technical Safeguards (§164.312)
- ✅ **Access Control**: Password-protected encryption
- ✅ **Encryption**: AES-256 (NIST approved algorithm)
- ✅ **Integrity Controls**: Verification step ensures data integrity

### Administrative Safeguards (§164.308)
- ✅ **Security Management**: Documented procedures (this README)
- ✅ **Assigned Security Responsibility**: Clear owner of key file
- ✅ **Workforce Security**: Access limited to authorized users
- ✅ **Information Access Management**: Password controls access

### Physical Safeguards (§164.310)
- ✅ **Device and Media Controls**: Secure storage requirements documented
- ✅ **Backup**: Multiple copies in secure locations

---

## Emergency Procedures

### If Password Is Lost
1. Accept that data cannot be re-identified
2. Document incident per institutional policy
3. Contact IRB to report situation
4. Update study documentation
5. Continue with permanently anonymized data

### If Encrypted File Is Lost
1. Check ALL backup locations immediately
2. Contact IT for backup restoration
3. If all copies lost: Same as password loss above
4. Document incident and lessons learned

### If Security Breach Suspected
1. **IMMEDIATELY** change password (create new encrypted file with new password)
2. Document breach details
3. Contact IRB
4. Contact institutional data security officer
5. Assess if PHI was actually compromised
6. Follow breach notification procedures if required

---

## Integration with De-identification Pipeline

The encrypted key file integrates with your comprehensive de-identification pipeline:

```
Workflow:
1. Decrypt key temporarily → deid_key.csv
2. Run: python3 comprehensive_deid_pipeline.py
3. Pipeline reads deid_key.csv
4. De-identification completes
5. IMMEDIATELY delete deid_key.csv
6. Only encrypted version remains
```

**Or modify the pipeline to read directly from encrypted file** (see Option B above).

---

## File Summary

After encryption, you should have:

```
/Users/peterpressman/Desktop/
├── deid_key_ENCRYPTED.zip          ← KEEP - encrypted key file
├── KEY_FILE_INSTRUCTIONS.txt       ← READ - critical security info
├── encrypt_key_file.py             ← Script to encrypt
├── decrypt_key.py                  ← Script to decrypt
└── deid_key.csv                    ← DELETE after encryption verified
```

**Backup locations should have:**
```
/Volumes/BackupDrive/SecureKeys/
└── deid_key_ENCRYPTED.zip          ← Backup copy 1

/Volumes/Nexus/ResearchKeys/
└── deid_key_ENCRYPTED.zip          ← Backup copy 2

~/Documents/SecureVault/
└── deid_key_ENCRYPTED.zip          ← Backup copy 3
```

---

## Questions?

### "Is AES-256 really HIPAA compliant?"
**Yes.** AES-256 is explicitly listed in NIST guidelines and widely accepted as HIPAA-compliant encryption.

### "Can I use a different encryption method?"
**Yes,** but ensure it meets HIPAA requirements:
- NIST-approved algorithm
- Minimum 128-bit keys (256-bit recommended)
- Document why chosen method meets requirements

### "How long should I keep the encrypted key?"
**Follow your institutional retention policy,** typically:
- Study duration + 7 years minimum
- Check IRB requirements
- May be longer for clinical trials

### "Should I encrypt the de-identified data too?"
**Good practice, but different requirement:**
- De-identified data (without key) = not PHI under HIPAA
- Encrypted key = necessary (contains PHI mappings)
- Still follow institutional data security policies

### "What if I need to add more patients later?"
1. Decrypt key file
2. Add new MRNs with UIDs and date shifts
3. Save updated CSV
4. Re-encrypt with SAME password
5. Delete unencrypted version
6. Update backups

---

## Final Reminders

🔑 **The encrypted key file is THE ONLY WAY to re-identify your data**

🔐 **Without password, recovery is IMPOSSIBLE**

💾 **Multiple backups in separate locations are CRITICAL**

🗑️ **Delete unencrypted copies immediately after encryption**

✅ **Test decryption quarterly to ensure everything works**

📝 **Document all of this in your IRB records**

---

**You're now ready to encrypt your key file with HIPAA-compliant security!**
