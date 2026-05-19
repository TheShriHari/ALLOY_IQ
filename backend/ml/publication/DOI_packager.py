import os
import shutil
import zipfile
from loguru import logger

class DOIPackager:
    """
    Stages scholarly datasets and metadata into a Zenodo/Figshare publication directory structure,
    and archives the target directories into a ZIP bundle ready for permanent DOI indexing.
    """
    def __init__(self, workspace_root: str):
        self.workspace_root = workspace_root

    def create_release_package(self, package_dir: str = "publication_release") -> str:
        """Assembles structured release staging directories."""
        package_path = os.path.join(self.workspace_root, package_dir)
        
        # Fresh directory setup
        if os.path.exists(package_path):
            shutil.rmtree(package_path)
            
        os.makedirs(package_path, exist_ok=True)

        # Stage subdirectories
        subdirs = ["dataset", "models", "configs", "reports", "manifests"]
        for sd in subdirs:
            os.makedirs(os.path.join(package_path, sd), exist_ok=True)
            
        logger.info("Initialized staged DOI release structure at: {}", package_path)
        return package_path

    def stage_file_to_package(self, src_path: str, package_dir: str, subfolder: str) -> str:
        """Copies a target validation asset into the appropriate package subdirectory."""
        if not os.path.exists(src_path):
            logger.warning("Source file not found for staging: {}", src_path)
            return ""
            
        dest_folder = os.path.join(self.workspace_root, package_dir, subfolder)
        os.makedirs(dest_folder, exist_ok=True)
        
        dest_path = os.path.join(dest_folder, os.path.basename(src_path))
        shutil.copy2(src_path, dest_path)
        logger.info("Staged file: {} -> {}", src_path, dest_path)
        return dest_path

    def archive_release_package(self, package_dir: str = "publication_release") -> str:
        """Compresses the staged release directory into a zip archive."""
        package_path = os.path.join(self.workspace_root, package_dir)
        archive_zip = f"{package_path}.zip"
        
        if os.path.exists(archive_zip):
            os.remove(archive_zip)
            
        with zipfile.ZipFile(archive_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(package_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, start=package_path)
                    zipf.write(file_path, arcname=arcname)
                    
        logger.info("Created compressed publication ZIP archive at: {}", archive_zip)
        return archive_zip
