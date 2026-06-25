"""
Download data files required by stmods.

Run once after cloning:
    python setup_data.py

What this downloads
-------------------
MIST v1.2 stellar evolution tracks  (~107 MB, solar metallicity, vvcrit=0.4)
    Source: https://mist.science
    Used for: stellar masses > 1.4 Msol in stellar_evolution.py

Atmosphere models (Phoenix, ck04, k93) are fetched on demand at runtime
via stsynphot from the STScI TRDS server — no manual download needed.
BHAC15 pre-main-sequence tracks are already included in the repo (BHAC15_tracks.csv).
"""

import os
import sys
import tarfile
import urllib.request
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))

MIST_DIR  = 'MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS'
MIST_URL  = ('https://mist.science/data/tarballs_v1.2/'
             'MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS.txz')
MIST_SIZE_MB = 107


def _progress_hook(count, block_size, total_size):
    pct = min(count * block_size * 100 // total_size, 100)
    bar = '#' * (pct // 2)
    sys.stdout.write(f'\r  [{bar:<50}] {pct:3d}%')
    sys.stdout.flush()


def download_mist(dest_dir=HERE, force=False):
    """Download and extract the MIST EEP track tarball."""
    target = os.path.join(dest_dir, MIST_DIR)

    if os.path.isdir(target) and not force:
        eep_files = [f for f in os.listdir(target) if f.endswith('.track.eep')]
        if eep_files:
            print(f'MIST tracks already present at {target}  ({len(eep_files)} files).')
            return
        print(f'MIST directory exists but appears empty — re-downloading.')

    archive = os.path.join(dest_dir, MIST_DIR + '.txz')

    print(f'Downloading MIST v1.2 tracks (~{MIST_SIZE_MB} MB) ...')
    print(f'  URL: {MIST_URL}')
    try:
        urllib.request.urlretrieve(MIST_URL, archive, reporthook=_progress_hook)
        print()
    except Exception as e:
        print(f'\nDownload failed: {e}')
        print('You can download the file manually from:')
        print(f'  {MIST_URL}')
        print(f'and extract it into:  {dest_dir}')
        if os.path.exists(archive):
            os.remove(archive)
        return

    print(f'Extracting {os.path.basename(archive)} ...')
    try:
        with tarfile.open(archive, 'r:xz') as tf:
            tf.extractall(dest_dir)
        print(f'Extracted to {target}')
    except Exception as e:
        print(f'Extraction failed: {e}')
        return
    finally:
        if os.path.exists(archive):
            os.remove(archive)

    eep_files = [f for f in os.listdir(target) if f.endswith('.track.eep')]
    print(f'Done — {len(eep_files)} MIST track files available.')


def check_dependencies():
    """Report which optional packages are available."""
    print('\nChecking Python dependencies ...')
    packages = {
        'numpy':      'required',
        'scipy':      'required',
        'pandas':     'required',
        'matplotlib': 'required',
        'astropy':    'required',
        'stsynphot':  'required for atmosphere models (auto-fetched from STScI)',
        'synphot':    'required by stsynphot',
        'pysynphot':  'optional — legacy; stsynphot is used instead',
    }
    all_ok = True
    for pkg, note in packages.items():
        try:
            __import__(pkg)
            print(f'  {pkg:12s}  OK   ({note})')
        except ImportError:
            required = note.startswith('required')
            marker = 'MISSING' if required else 'absent '
            print(f'  {pkg:12s}  {marker}  ({note})')
            if required:
                all_ok = False

    if not all_ok:
        print('\nInstall missing packages with:')
        print('  pip install numpy scipy pandas matplotlib astropy stsynphot synphot')
    return all_ok


def check_mist(dest_dir=HERE):
    """Return True if MIST tracks are present and non-empty."""
    target = os.path.join(dest_dir, MIST_DIR)
    if not os.path.isdir(target):
        return False
    return any(f.endswith('.track.eep') for f in os.listdir(target))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--force', action='store_true',
                        help='Re-download even if data already exists')
    parser.add_argument('--check-only', action='store_true',
                        help='Only report status, do not download')
    args = parser.parse_args()

    ok = check_dependencies()

    print('\nChecking data files ...')
    mist_ok = check_mist()
    if mist_ok:
        n = sum(1 for f in os.listdir(os.path.join(HERE, MIST_DIR))
                if f.endswith('.track.eep'))
        print(f'  MIST tracks   OK   ({n} files in {MIST_DIR}/)')
    else:
        print(f'  MIST tracks   MISSING  ({MIST_DIR}/ not found)')

    print(f'  BHAC15 tracks OK   (BHAC15_tracks.csv — included in repo)')
    print(f'  Atmosphere models  fetched at runtime from STScI TRDS (internet required)')

    if not args.check_only:
        if not mist_ok or args.force:
            print()
            download_mist(dest_dir=HERE, force=args.force)
        else:
            print('\nAll data present. Run with --force to re-download.')
