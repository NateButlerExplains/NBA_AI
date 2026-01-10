#!/usr/bin/env python3
"""
setup.py - NBA AI Project Setup Script

This script automates the complete setup process for new users:
1. Creates a Python virtual environment
2. Installs all dependencies
3. Downloads the current season database from GitHub Releases
4. Downloads trained model files from GitHub Releases
5. Creates .env configuration file
6. Verifies the installation works

Usage:
    python setup.py              # Full setup (recommended for new users)
    python setup.py --skip-venv  # Skip venv creation (use existing)
    python setup.py --skip-data  # Skip data/model download
    python setup.py --skip-deps  # Skip dependency installation

Requirements:
    - Python 3.10 or higher
    - Internet connection (for downloads)
    - ~2GB disk space (database + models + dependencies)
"""

import argparse
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

GITHUB_REPO = "NBA-Betting/NBA_AI"
RELEASE_TAG = "v0.4.0"  # Release tag for v0.4.0

# Files to download from GitHub Releases
DATABASE_ZIP_FILENAME = "NBA_AI_current.zip"  # Zipped database (~43MB)
DATABASE_FILENAME = "NBA_AI_current.sqlite"  # Extracted database (~668MB)
MODELS_FILENAME = "models_v0.4.zip"  # Trained ML models v0.4 (~273KB)

# Local fallback for testing (set via --local-source=/path/to/NBA_AI)
LOCAL_SOURCE = None

# Directory structure
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
VENV_DIR = PROJECT_ROOT / "venv"

# =============================================================================
# Helper Functions
# =============================================================================


def print_banner():
    """Print the setup banner."""
    print("\n" + "=" * 60)
    print("  🏀 NBA AI - Project Setup")
    print("=" * 60)
    print("\nThis script will set up everything you need to run NBA AI.")
    print("It may take a few minutes to download dependencies and data.\n")


def print_step(msg):
    """Print a formatted step message."""
    print(f"\n{'─'*60}")
    print(f"  📦 {msg}")
    print(f"{'─'*60}\n")


def print_success(msg):
    """Print a success message."""
    print(f"  ✅ {msg}")


def print_warning(msg):
    """Print a warning message."""
    print(f"  ⚠️  {msg}")


def print_error(msg):
    """Print an error message."""
    print(f"  ❌ {msg}")


def print_info(msg):
    """Print an info message."""
    print(f"  ℹ️  {msg}")


def run_command(cmd, check=True, capture=True):
    """Run a shell command and return the result."""
    if capture:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    else:
        result = subprocess.run(cmd, shell=True)

    if check and result.returncode != 0:
        print_error(f"Command failed: {cmd}")
        if capture and result.stderr:
            print(f"      {result.stderr.strip()}")
        return None
    return result


def get_venv_python():
    """Get the path to the venv Python executable."""
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def get_venv_pip():
    """Get the path to the venv pip executable."""
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "pip.exe"
    return VENV_DIR / "bin" / "pip"


def get_activate_command():
    """Get the command to activate the virtual environment."""
    if platform.system() == "Windows":
        return r"venv\Scripts\activate"
    return "source venv/bin/activate"


# =============================================================================
# Setup Steps
# =============================================================================


def check_python_version():
    """Ensure Python 3.10+ is being used."""
    print_step("Step 1/6: Checking Python version")

    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print_error(f"Python 3.10+ required. You have: {version_str}")
        print_info("Please install Python 3.10 or higher and try again.")
        print_info("Download from: https://www.python.org/downloads/")
        sys.exit(1)

    print_success(f"Python {version_str} detected")
    return True


def create_virtual_environment():
    """Create a Python virtual environment."""
    print_step("Step 2/6: Creating virtual environment")

    if VENV_DIR.exists():
        print_warning("Virtual environment already exists")
        response = input("      Recreate it? (y/N): ").strip().lower()
        if response == "y":
            print_info("Removing existing virtual environment...")
            shutil.rmtree(VENV_DIR)
        else:
            print_info("Using existing virtual environment")
            return True

    print_info("Creating virtual environment (this may take a moment)...")
    result = run_command(f'"{sys.executable}" -m venv "{VENV_DIR}"')

    if result is None:
        print_error("Failed to create virtual environment")
        print_info("Try running: python -m venv venv")
        return False

    print_success(f"Virtual environment created at: venv/")
    return True


def install_dependencies():
    """Install Python dependencies from requirements.txt."""
    print_step("Step 3/6: Installing dependencies")

    pip_path = get_venv_pip()
    requirements_file = PROJECT_ROOT / "requirements.txt"

    if not requirements_file.exists():
        print_error("requirements.txt not found")
        return False

    # Upgrade pip first (quietly)
    print_info("Upgrading pip...")
    run_command(f'"{pip_path}" install --upgrade pip -q', check=False)

    # Install requirements
    print_info("Installing packages (this may take several minutes)...")
    print_info("Installing: Flask, NumPy, Pandas, scikit-learn, XGBoost...")

    result = run_command(
        f'"{pip_path}" install -r "{requirements_file}"', capture=False
    )

    if result is None or result.returncode != 0:
        print_error("Failed to install dependencies")
        print_info(f"Try running: {pip_path} install -r requirements.txt")
        return False

    print_success("All dependencies installed")
    return True


def download_file_with_progress(url, dest_path, description="file"):
    """Download a file with progress indication."""
    print_info(f"Downloading {description}...")
    print_info(f"  URL: {url}")

    try:

        def reporthook(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 // total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(
                    f"\r      Progress: {percent:3d}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)",
                    end="",
                    flush=True,
                )

        urllib.request.urlretrieve(url, dest_path, reporthook)
        print()  # Newline after progress
        return True
    except urllib.error.HTTPError as e:
        print()
        print_error(f"Download failed: HTTP {e.code}")
        if e.code == 404:
            print_info("The release files may not be uploaded yet.")
            print_info(f"Check: https://github.com/{GITHUB_REPO}/releases")
        return False
    except Exception as e:
        print()
        print_error(f"Download failed: {e}")
        return False


def copy_file_with_progress(src_path, dest_path, description="file"):
    """Copy a file from local source with progress indication."""
    print_info(f"Copying {description} from local source...")
    print_info(f"  Source: {src_path}")

    try:
        total_size = src_path.stat().st_size
        copied = 0
        chunk_size = 1024 * 1024  # 1MB chunks

        with open(src_path, "rb") as src, open(dest_path, "wb") as dst:
            while True:
                chunk = src.read(chunk_size)
                if not chunk:
                    break
                dst.write(chunk)
                copied += len(chunk)
                percent = min(100, copied * 100 // total_size)
                mb_copied = copied / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(
                    f"\r      Progress: {percent:3d}% ({mb_copied:.1f}/{mb_total:.1f} MB)",
                    end="",
                    flush=True,
                )
        print()  # Newline after progress
        return True
    except Exception as e:
        print()
        print_error(f"Copy failed: {e}")
        return False


def get_release_url(filename):
    """Get the download URL for a release asset."""
    return (
        f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}/{filename}"
    )


def download_data_files():
    """Download required data files from GitHub Releases or copy from local source."""
    print_step("Step 4/6: Downloading data files")

    # Ensure directories exist
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    success = True
    use_local = LOCAL_SOURCE is not None

    if use_local:
        print_info(f"Using local source: {LOCAL_SOURCE}")

    # Download/copy database
    db_dest = DATA_DIR / DATABASE_FILENAME
    if db_dest.exists():
        size_mb = db_dest.stat().st_size / (1024 * 1024)
        print_warning(f"Database already exists: {db_dest.name} ({size_mb:.1f} MB)")
        response = input("      Redownload? (y/N): ").strip().lower()
        if response == "y":
            db_dest.unlink()
        else:
            print_info("Keeping existing database")

    if not db_dest.exists():
        if use_local:
            # Copy from local source
            local_db = Path(LOCAL_SOURCE) / "data" / DATABASE_FILENAME
            if local_db.exists():
                if copy_file_with_progress(local_db, db_dest, "database"):
                    size_mb = db_dest.stat().st_size / (1024 * 1024)
                    print_success(f"Database copied: {db_dest.name} ({size_mb:.1f} MB)")
                else:
                    success = False
            else:
                print_error(f"Local database not found: {local_db}")
                success = False
        else:
            # Download zip from GitHub Releases and extract
            db_zip_dest = DATA_DIR / DATABASE_ZIP_FILENAME
            url = get_release_url(DATABASE_ZIP_FILENAME)
            if not download_file_with_progress(url, db_zip_dest, "database"):
                print_warning("Database download failed")
                print_info("You can download manually from GitHub Releases:")
                print_info(
                    f"  https://github.com/{GITHUB_REPO}/releases/tag/{RELEASE_TAG}"
                )
                success = False
            else:
                # Extract the zip
                print_info("Extracting database...")
                try:
                    with zipfile.ZipFile(db_zip_dest, "r") as zip_ref:
                        zip_ref.extractall(DATA_DIR)
                    db_zip_dest.unlink()  # Remove zip after extraction
                    size_mb = db_dest.stat().st_size / (1024 * 1024)
                    print_success(
                        f"Database extracted: {db_dest.name} ({size_mb:.1f} MB)"
                    )
                except Exception as e:
                    print_error(f"Failed to extract database: {e}")
                    success = False

    # Download/copy models
    has_models = any(MODELS_DIR.glob("*.joblib")) or any(MODELS_DIR.glob("*.pth"))

    if has_models:
        print_warning("Model files already exist")
        response = input("      Redownload? (y/N): ").strip().lower()
        if response == "y":
            for f in MODELS_DIR.glob("*.joblib"):
                f.unlink()
            for f in MODELS_DIR.glob("*.pth"):
                f.unlink()
            for f in MODELS_DIR.glob("*.json"):
                f.unlink()
            has_models = False
        else:
            print_info("Keeping existing models")

    if not has_models:
        models_zip = PROJECT_ROOT / MODELS_FILENAME  # Download to project root

        if use_local:
            # Copy from local source (models are in data/releases/v0.4/)
            local_models_zip = Path(LOCAL_SOURCE) / "data" / "releases" / "v0.4" / MODELS_FILENAME
            print_info(f"Looking for models at: {local_models_zip}")
            if local_models_zip.exists():
                if copy_file_with_progress(local_models_zip, models_zip, "models"):
                    print_info("Extracting models...")
                    try:
                        with zipfile.ZipFile(models_zip, "r") as zip_ref:
                            zip_ref.extractall(MODELS_DIR)  # Extract to models directory
                        models_zip.unlink()  # Remove zip after extraction
                        print_success("Models extracted successfully")
                    except Exception as e:
                        print_error(f"Failed to extract models: {e}")
                        success = False
                else:
                    success = False
            else:
                print_error(f"Local models.zip not found: {local_models_zip}")
                success = False
        else:
            # Download from GitHub Releases
            url = get_release_url(MODELS_FILENAME)

            if download_file_with_progress(url, models_zip, "models"):
                print_info("Extracting models...")
                try:
                    with zipfile.ZipFile(models_zip, "r") as zip_ref:
                        zip_ref.extractall(MODELS_DIR)  # Extract to models directory
                    models_zip.unlink()  # Remove zip after extraction
                    print_success("Models extracted successfully")
                except Exception as e:
                    print_error(f"Failed to extract models: {e}")
                    success = False
            else:
                print_error("Models download failed")
                print_info("Please download models manually from GitHub Releases")
                success = False

    return success


def create_env_file():
    """Create .env file from .env.example if it doesn't exist."""
    print_step("Step 5/6: Configuring environment")

    env_file = PROJECT_ROOT / ".env"
    env_example = PROJECT_ROOT / ".env.example"

    if env_file.exists():
        print_warning(".env file already exists")
        response = input("      Overwrite? (y/N): ").strip().lower()
        if response != "y":
            print_info("Keeping existing .env file")
            return True
        env_file.unlink()

    # Determine database path
    db_path = f"data/{DATABASE_FILENAME}"

    if env_example.exists():
        # Copy and customize from example
        content = env_example.read_text()

        # Update database path
        content = content.replace(
            "DATABASE_PATH=data/NBA_AI_current.sqlite", f"DATABASE_PATH={db_path}"
        )

        # Uncomment and set PROJECT_ROOT
        lines = content.split("\n")
        new_lines = []
        for line in lines:
            if line.strip().startswith("# PROJECT_ROOT="):
                new_lines.append(f"PROJECT_ROOT={PROJECT_ROOT}")
            else:
                new_lines.append(line)

        env_file.write_text("\n".join(new_lines))
        print_success(".env file created from .env.example")
    else:
        # Create minimal .env
        env_content = f"""# NBA AI Environment Configuration
# Generated by setup.py

PROJECT_ROOT={PROJECT_ROOT}
DATABASE_PATH={db_path}
"""
        env_file.write_text(env_content)
        print_success(".env file created")

    return True


def verify_installation():
    """Verify the installation is working."""
    print_step("Step 6/6: Verifying installation")

    python_path = get_venv_python()
    all_good = True

    # Test 1: Check config loads and get default predictor
    print_info("Testing configuration...")
    result = run_command(
        f'"{python_path}" -c "from src.config import config; '
        f"print(config['database']['path'], config.get('default_predictor', 'Tree'))\"",
        check=False,
    )
    default_predictor = "Tree"  # fallback
    if result and result.returncode == 0:
        parts = result.stdout.strip().split()
        db_config_path = parts[0] if parts else ""
        default_predictor = parts[1] if len(parts) > 1 else "Tree"
        print_success(f"Configuration loads: {db_config_path}")
        print_info(f"Default predictor: {default_predictor}")
    else:
        print_error("Configuration failed to load")
        all_good = False

    # Test 2: Check database exists and has data
    print_info("Testing database...")
    db_path = DATA_DIR / DATABASE_FILENAME
    if db_path.exists():
        result = run_command(
            f'"{python_path}" -c "import sqlite3; conn=sqlite3.connect(\'{db_path}\'); '
            f"c=conn.cursor(); c.execute('SELECT COUNT(*) FROM Games'); print(c.fetchone()[0])\"",
            check=False,
        )
        if result and result.returncode == 0:
            count = result.stdout.strip()
            print_success(f"Database accessible: {count} games found")
        else:
            print_warning("Database exists but couldn't query it")
    else:
        print_error(f"Database not found at: {db_path}")
        all_good = False

    # Test 3: Check model files (required for Tree and Linear predictors)
    print_info("Checking model files...")
    xgb_model = MODELS_DIR / "xgboost_v0.4_mae10.1.joblib"
    ridge_model = MODELS_DIR / "ridge_v0.4_mae11.2.joblib"

    if xgb_model.exists() and ridge_model.exists():
        print_success("All model files present (Tree and Linear predictors ready)")
    else:
        missing = []
        if not xgb_model.exists():
            missing.append("XGBoost (Tree predictor)")
        if not ridge_model.exists():
            missing.append("Ridge (Linear predictor)")
        print_error(f"Missing models: {', '.join(missing)}")
        print_info("Please re-run setup.py or download models from GitHub Releases")
        all_good = False

    # Test 4: Test all available predictors
    print_info("Testing predictors...")
    # Get a game_id for an upcoming/scheduled game (status=1)
    get_game_id_cmd = (
        f'"{python_path}" -c "import sqlite3; '
        f"conn=sqlite3.connect('{db_path}'); c=conn.cursor(); "
        f"c.execute('SELECT game_id FROM Games WHERE status = 1 ORDER BY date_time_utc LIMIT 1'); "
        f"result = c.fetchone(); print(result[0] if result else '')\""
    )
    game_id_result = run_command(get_game_id_cmd, check=False)
    test_game_id = (
        game_id_result.stdout.strip()
        if game_id_result and game_id_result.returncode == 0 and game_id_result.stdout.strip()
        else None
    )

    if test_game_id:
        # Test all three predictors using a temp script to handle imports cleanly
        test_script = PROJECT_ROOT / "_test_predictors.py"
        test_script.write_text(f'''
import sys
from src.config import config

test_game_id = "{test_game_id}"
results = []

# Test Baseline (no model needed)
try:
    from src.predictions.prediction_engines.baseline_predictor import BaselinePredictor
    p = BaselinePredictor()
    r = p.make_pre_game_predictions([test_game_id])
    results.append(("Baseline", test_game_id in r))
except Exception as e:
    results.append(("Baseline", False))

# Test Linear (needs model_paths from config)
try:
    from src.predictions.prediction_engines.linear_predictor import LinearPredictor
    model_paths = config.get("predictors", {{}}).get("Linear", {{}}).get("model_paths", [])
    if model_paths:
        p = LinearPredictor(model_paths=model_paths)
        r = p.make_pre_game_predictions([test_game_id])
        results.append(("Linear", test_game_id in r))
    else:
        results.append(("Linear", False))
except Exception as e:
    results.append(("Linear", False))

# Test Tree (needs model_paths from config)
try:
    from src.predictions.prediction_engines.tree_predictor import TreePredictor
    model_paths = config.get("predictors", {{}}).get("Tree", {{}}).get("model_paths", [])
    if model_paths:
        p = TreePredictor(model_paths=model_paths)
        r = p.make_pre_game_predictions([test_game_id])
        results.append(("Tree", test_game_id in r))
    else:
        results.append(("Tree", False))
except Exception as e:
    results.append(("Tree", False))

# Output results
ok = [name for name, success in results if success]
fail = [name for name, success in results if not success]
print("OK:" + ",".join(ok) if ok else "OK:")
print("FAIL:" + ",".join(fail) if fail else "FAIL:")
''')
        result = run_command(f'"{python_path}" "{test_script}"', check=False)
        test_script.unlink()  # Clean up

        if result and result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            ok_line = [l for l in lines if l.startswith("OK:")][0] if lines else "OK:"
            fail_line = [l for l in lines if l.startswith("FAIL:")][0] if lines else "FAIL:"
            predictors_ok = [p for p in ok_line.replace("OK:", "").split(",") if p]
            predictors_failed = [p for p in fail_line.replace("FAIL:", "").split(",") if p]

            if predictors_ok:
                print_success(f"Predictors working: {', '.join(predictors_ok)}")
            if predictors_failed:
                print_error(f"Predictors failed: {', '.join(predictors_failed)}")
                all_good = False
        else:
            print_error("Could not test predictors")
            all_good = False
    else:
        print_warning("No upcoming games in database to test predictors")
        print_info("Predictors will work when schedule data is updated")

    # Test 5: Check Flask app with default predictor
    print_info(f"Testing web app with {default_predictor} predictor...")
    test_script = PROJECT_ROOT / "_test_flask.py"
    test_script.write_text(
        f"from src.web_app.app import create_app\n"
        f"app = create_app(predictor='{default_predictor}')\n"
        f"print('OK' if app else 'FAIL')\n"
    )
    result = run_command(f'"{python_path}" "{test_script}"', check=False)
    test_script.unlink()  # Clean up
    if result and result.returncode == 0 and "OK" in result.stdout:
        print_success(f"Flask app creates successfully with {default_predictor} predictor")
    else:
        print_error("Flask app test failed")
        if result and result.stderr:
            print_info(f"Error: {result.stderr.strip()[:200]}")
        all_good = False

    return all_good


def print_completion_message(success):
    """Print the completion message with next steps."""
    print("\n" + "=" * 60)

    if success:
        print("  🎉 Setup Complete!")
    else:
        print("  ⚠️  Setup completed with warnings")

    print("=" * 60)

    activate_cmd = get_activate_command()

    print(
        f"""
┌─────────────────────────────────────────────────────────────┐
│  Next Steps:                                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Activate the virtual environment:                       │
│     {activate_cmd:<43} │
│                                                             │
│  2. Start the web app:                                      │
│     python start_app.py                                     │
│                                                             │
│  3. Open your browser to:                                   │
│     http://localhost:5000                                   │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  Optional:                                                  │
│  • Run with debug mode: python start_app.py --debug         │
│  • Use different predictor: python start_app.py --predictor=Linear │
│  • Run tests: python -m pytest tests/ -v                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘

For more information, see README.md
"""
    )


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    global LOCAL_SOURCE

    parser = argparse.ArgumentParser(
        description="NBA AI Project Setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup.py                 Full setup (recommended)
  python setup.py --skip-data     Skip downloading data files
  python setup.py --skip-venv     Use existing virtual environment
  python setup.py --local-source=/path/to/dev  Copy data from local source
        """,
    )
    parser.add_argument(
        "--skip-venv", action="store_true", help="Skip virtual environment creation"
    )
    parser.add_argument(
        "--skip-data", action="store_true", help="Skip data file downloads"
    )
    parser.add_argument(
        "--skip-deps", action="store_true", help="Skip dependency installation"
    )
    parser.add_argument(
        "--local-source",
        type=str,
        help="Copy data/models from local directory instead of downloading",
    )
    args = parser.parse_args()

    # Set local source globally
    if args.local_source:
        LOCAL_SOURCE = Path(args.local_source).absolute()
        if not LOCAL_SOURCE.exists():
            print_error(f"Local source directory not found: {LOCAL_SOURCE}")
            sys.exit(1)

    print_banner()

    # Run setup steps
    success = True

    if not check_python_version():
        sys.exit(1)

    if not args.skip_venv:
        if not create_virtual_environment():
            sys.exit(1)
    else:
        print_step("Step 2/6: Skipping virtual environment (--skip-venv)")

    if not args.skip_deps:
        if not install_dependencies():
            success = False
    else:
        print_step("Step 3/6: Skipping dependencies (--skip-deps)")

    if not args.skip_data:
        if not download_data_files():
            success = False
    else:
        print_step("Step 4/6: Skipping data download (--skip-data)")

    create_env_file()

    if not verify_installation():
        success = False

    print_completion_message(success)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
