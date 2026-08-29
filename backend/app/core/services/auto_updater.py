"""
Auto-Update System
Checks for updates from GitHub and manages update installation
"""
import os
import json
import requests
import subprocess
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Configuration
GITHUB_REPO = os.getenv("GITHUB_REPO", "yourusername/fin-dash-buddy")
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
UPDATE_CHECK_FILE = Path.home() / ".findashbuddy" / "last_update_check.json"
CURRENT_VERSION = os.getenv("APP_VERSION", "1.0.0")


class AutoUpdater:
    """Handles automatic updates from GitHub"""
    
    def __init__(self):
        self.update_dir = Path.home() / ".findashbuddy" / "updates"
        self.update_dir.mkdir(parents=True, exist_ok=True)
        UPDATE_CHECK_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    def check_for_updates(self, force: bool = False) -> Optional[Dict[str, Any]]:
        """
        Check if updates are available
        
        Args:
            force: Force check even if recently checked
            
        Returns:
            Update info dict if available, None otherwise
        """
        # Check if we should skip (recently checked)
        if not force and not self._should_check():
            logger.info("Skipping update check (recently checked)")
            return None
        
        try:
            # Fetch latest release from GitHub
            response = requests.get(GITHUB_API_URL, timeout=10)
            response.raise_for_status()
            
            release_data = response.json()
            latest_version = release_data['tag_name'].lstrip('v')
            
            # Save check time
            self._save_check_time()
            
            # Compare versions
            if self._is_newer_version(latest_version, CURRENT_VERSION):
                update_info = {
                    "available": True,
                    "current_version": CURRENT_VERSION,
                    "latest_version": latest_version,
                    "release_url": release_data['html_url'],
                    "release_notes": release_data.get('body', ''),
                    "published_at": release_data['published_at'],
                    "assets": [
                        {
                            "name": asset['name'],
                            "download_url": asset['browser_download_url'],
                            "size": asset['size']
                        }
                        for asset in release_data.get('assets', [])
                    ]
                }
                
                logger.info(f"Update available: {CURRENT_VERSION} -> {latest_version}")
                return update_info
            else:
                logger.info(f"Already on latest version: {CURRENT_VERSION}")
                return {"available": False, "current_version": CURRENT_VERSION}
                
        except Exception as e:
            logger.error(f"Failed to check for updates: {e}")
            return None
    
    def download_update(self, update_info: Dict[str, Any]) -> Optional[Path]:
        """
        Download update package
        
        Args:
            update_info: Update information from check_for_updates
            
        Returns:
            Path to downloaded file or None
        """
        try:
            # Find the appropriate asset for current platform
            import platform
            system = platform.system().lower()
            
            asset = None
            for a in update_info.get('assets', []):
                if system in a['name'].lower():
                    asset = a
                    break
            
            if not asset:
                logger.error("No compatible update package found")
                return None
            
            # Download
            logger.info(f"Downloading update: {asset['name']}")
            response = requests.get(asset['download_url'], stream=True, timeout=30)
            response.raise_for_status()
            
            # Save to file
            download_path = self.update_dir / asset['name']
            with open(download_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Update downloaded to: {download_path}")
            return download_path
            
        except Exception as e:
            logger.error(f"Failed to download update: {e}")
            return None
    
    def verify_update(self, update_path: Path, expected_checksum: Optional[str] = None) -> bool:
        """
        Verify update package integrity
        
        Args:
            update_path: Path to update file
            expected_checksum: Expected SHA256 checksum (optional)
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Calculate checksum
            sha256_hash = hashlib.sha256()
            with open(update_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            
            actual_checksum = sha256_hash.hexdigest()
            
            if expected_checksum:
                if actual_checksum != expected_checksum:
                    logger.error(f"Checksum mismatch: {actual_checksum} != {expected_checksum}")
                    return False
            
            logger.info(f"Update verified: {actual_checksum}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to verify update: {e}")
            return False
    
    def install_update(self, update_path: Path) -> bool:
        """
        Install update (Docker-based)
        
        Args:
            update_path: Path to update package
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # For Docker deployment, we just need to pull latest and restart
            logger.info("Installing update via Docker...")
            
            # Run update script
            project_root = Path(__file__).parent.parent.parent.parent
            update_script = project_root / "deploy.sh"
            
            if update_script.exists():
                result = subprocess.run(
                    [str(update_script), "update"],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    logger.info("Update installed successfully")
                    return True
                else:
                    logger.error(f"Update failed: {result.stderr}")
                    return False
            else:
                logger.error("Update script not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to install update: {e}")
            return False
    
    def _should_check(self) -> bool:
        """Check if we should check for updates (not checked recently)"""
        if not UPDATE_CHECK_FILE.exists():
            return True
        
        try:
            with open(UPDATE_CHECK_FILE, 'r') as f:
                data = json.load(f)
            
            last_check = datetime.fromisoformat(data['last_check'])
            check_interval = timedelta(hours=24)  # Check once per day
            
            return datetime.now() - last_check > check_interval
            
        except:
            return True
    
    def _save_check_time(self):
        """Save the time of last update check"""
        try:
            with open(UPDATE_CHECK_FILE, 'w') as f:
                json.dump({
                    'last_check': datetime.now().isoformat(),
                    'current_version': CURRENT_VERSION
                }, f)
        except Exception as e:
            logger.error(f"Failed to save check time: {e}")
    
    def _is_newer_version(self, version1: str, version2: str) -> bool:
        """Compare semantic versions"""
        try:
            v1_parts = [int(x) for x in version1.split('.')]
            v2_parts = [int(x) for x in version2.split('.')]
            return v1_parts > v2_parts
        except:
            return False


# Singleton instance
auto_updater = AutoUpdater()
