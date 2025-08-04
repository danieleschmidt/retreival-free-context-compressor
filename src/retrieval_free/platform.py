"""Cross-platform compatibility utilities."""

import os
import sys
import platform
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import subprocess
import shutil

logger = logging.getLogger(__name__)


class PlatformInfo:
    """Information about the current platform."""
    
    def __init__(self):
        """Initialize platform information."""
        self._system = platform.system().lower()
        self._machine = platform.machine().lower()
        self._python_version = sys.version_info
        self._platform_details = self._get_platform_details()
    
    @property
    def system(self) -> str:
        """Get system name (linux, windows, darwin)."""
        return self._system
    
    @property
    def is_windows(self) -> bool:
        """Check if running on Windows."""
        return self._system == "windows"
    
    @property
    def is_linux(self) -> bool:
        """Check if running on Linux."""
        return self._system == "linux"
    
    @property
    def is_macos(self) -> bool:
        """Check if running on macOS."""
        return self._system == "darwin"
    
    @property
    def architecture(self) -> str:
        """Get system architecture."""
        return self._machine
    
    @property
    def is_64bit(self) -> bool:
        """Check if running on 64-bit architecture."""
        return self._machine in ["x86_64", "amd64", "arm64", "aarch64"]
    
    @property
    def python_version(self) -> Tuple[int, int, int]:
        """Get Python version tuple."""
        return (self._python_version.major, self._python_version.minor, self._python_version.micro)
    
    @property
    def python_version_string(self) -> str:
        """Get Python version as string."""
        return f"{self._python_version.major}.{self._python_version.minor}.{self._python_version.micro}"
    
    def _get_platform_details(self) -> Dict[str, Any]:
        """Get detailed platform information."""
        details = {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
        }
        
        # Add OS-specific details
        if self.is_linux:
            try:
                # Try to get Linux distribution info
                import distro
                details.update({
                    "distro_name": distro.name(),
                    "distro_version": distro.version(),
                    "distro_id": distro.id(),
                })
            except ImportError:
                # Fallback for older systems
                try:
                    with open("/etc/os-release", "r") as f:
                        os_release = f.read()
                        details["os_release"] = os_release
                except FileNotFoundError:
                    pass
        
        elif self.is_windows:
            details.update({
                "windows_edition": platform.win32_edition(),
                "windows_version": platform.win32_ver(),
            })
        
        elif self.is_macos:
            details.update({
                "mac_version": platform.mac_ver(),
            })
        
        return details
    
    def get_info_dict(self) -> Dict[str, Any]:
        """Get platform information as dictionary."""
        return {
            "system": self.system,
            "architecture": self.architecture,
            "is_64bit": self.is_64bit,
            "python_version": self.python_version_string,
            "platform_details": self._platform_details
        }


class PathManager:
    """Cross-platform path management."""
    
    def __init__(self):
        """Initialize path manager."""
        self.platform_info = PlatformInfo()
    
    def get_home_dir(self) -> Path:
        """Get user home directory."""
        return Path.home()
    
    def get_config_dir(self, app_name: str = "retrieval_free") -> Path:
        """Get application configuration directory.
        
        Args:
            app_name: Application name
            
        Returns:
            Configuration directory path
        """
        if self.platform_info.is_windows:
            # Windows: %APPDATA%\AppName
            config_dir = Path(os.environ.get("APPDATA", "")) / app_name
        elif self.platform_info.is_macos:
            # macOS: ~/Library/Application Support/AppName
            config_dir = self.get_home_dir() / "Library" / "Application Support" / app_name
        else:
            # Linux: ~/.config/AppName
            config_dir = Path(os.environ.get("XDG_CONFIG_HOME", self.get_home_dir() / ".config")) / app_name
        
        return config_dir
    
    def get_data_dir(self, app_name: str = "retrieval_free") -> Path:
        """Get application data directory.
        
        Args:
            app_name: Application name
            
        Returns:
            Data directory path
        """
        if self.platform_info.is_windows:
            # Windows: %LOCALAPPDATA%\AppName
            data_dir = Path(os.environ.get("LOCALAPPDATA", "")) / app_name
        elif self.platform_info.is_macos:
            # macOS: ~/Library/Application Support/AppName
            data_dir = self.get_home_dir() / "Library" / "Application Support" / app_name
        else:
            # Linux: ~/.local/share/AppName
            data_dir = Path(os.environ.get("XDG_DATA_HOME", self.get_home_dir() / ".local" / "share")) / app_name
        
        return data_dir
    
    def get_cache_dir(self, app_name: str = "retrieval_free") -> Path:
        """Get application cache directory.
        
        Args:
            app_name: Application name
            
        Returns:
            Cache directory path
        """
        if self.platform_info.is_windows:
            # Windows: %LOCALAPPDATA%\AppName\Cache
            cache_dir = Path(os.environ.get("LOCALAPPDATA", "")) / app_name / "Cache"
        elif self.platform_info.is_macos:
            # macOS: ~/Library/Caches/AppName
            cache_dir = self.get_home_dir() / "Library" / "Caches" / app_name
        else:
            # Linux: ~/.cache/AppName
            cache_dir = Path(os.environ.get("XDG_CACHE_HOME", self.get_home_dir() / ".cache")) / app_name
        
        return cache_dir
    
    def get_temp_dir(self) -> Path:
        """Get temporary directory."""
        import tempfile
        return Path(tempfile.gettempdir())
    
    def normalize_path(self, path: str) -> Path:
        """Normalize path for current platform.
        
        Args:
            path: Path string to normalize
            
        Returns:
            Normalized Path object
        """
        return Path(path).resolve()
    
    def ensure_dir(self, path: Path, mode: int = 0o755) -> bool:
        """Ensure directory exists with proper permissions.
        
        Args:
            path: Directory path
            mode: Directory permissions (Unix only)
            
        Returns:
            True if directory was created or already exists
        """
        try:
            path.mkdir(parents=True, exist_ok=True)
            
            # Set permissions on Unix-like systems
            if not self.platform_info.is_windows:
                path.chmod(mode)
            
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")
            return False


class ProcessManager:
    """Cross-platform process management."""
    
    def __init__(self):
        """Initialize process manager."""
        self.platform_info = PlatformInfo()
    
    def run_command(
        self,
        command: List[str],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        shell: bool = False
    ) -> Tuple[int, str, str]:
        """Run command cross-platform.
        
        Args:
            command: Command and arguments
            cwd: Working directory
            env: Environment variables
            timeout: Timeout in seconds
            shell: Whether to use shell
            
        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        try:
            # On Windows, some commands need shell=True
            if self.platform_info.is_windows and not shell:
                # Check if command is a built-in Windows command
                builtin_commands = ['dir', 'copy', 'move', 'del', 'type', 'echo']
                if command[0].lower() in builtin_commands:
                    shell = True
            
            result = subprocess.run(
                command,
                cwd=cwd,
                env=env,
                timeout=timeout,
                shell=shell,
                capture_output=True,
                text=True
            )
            
            return result.returncode, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except FileNotFoundError:
            return -1, "", f"Command not found: {command[0]}"
        except Exception as e:
            return -1, "", str(e)
    
    def find_executable(self, name: str) -> Optional[str]:
        """Find executable in PATH.
        
        Args:
            name: Executable name
            
        Returns:
            Full path to executable or None if not found
        """
        return shutil.which(name)
    
    def get_process_info(self) -> Dict[str, Any]:
        """Get current process information."""
        return {
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "cwd": str(Path.cwd()),
            "executable": sys.executable,
            "argv": sys.argv,
        }


class EnvironmentManager:
    """Cross-platform environment management."""
    
    def __init__(self):
        """Initialize environment manager."""
        self.platform_info = PlatformInfo()
        self.path_manager = PathManager()
    
    def get_env_var(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable value.
        
        Args:
            name: Variable name
            default: Default value if not found
            
        Returns:
            Environment variable value or default
        """
        return os.environ.get(name, default)
    
    def set_env_var(self, name: str, value: str) -> None:
        """Set environment variable.
        
        Args:
            name: Variable name
            value: Variable value
        """
        os.environ[name] = value
    
    def get_path_separator(self) -> str:
        """Get path separator for current platform."""
        return os.pathsep
    
    def get_path_env(self) -> List[str]:
        """Get PATH environment variable as list."""
        path_env = self.get_env_var("PATH", "")
        return path_env.split(self.get_path_separator()) if path_env else []
    
    def add_to_path(self, path: str) -> None:
        """Add directory to PATH environment variable.
        
        Args:
            path: Directory path to add
        """
        current_path = self.get_path_env()
        if path not in current_path:
            current_path.insert(0, path)
            new_path = self.get_path_separator().join(current_path)
            self.set_env_var("PATH", new_path)
    
    def get_python_path(self) -> List[str]:
        """Get PYTHONPATH as list."""
        python_path = self.get_env_var("PYTHONPATH", "")
        return python_path.split(self.get_path_separator()) if python_path else []
    
    def setup_environment(self) -> Dict[str, Any]:
        """Setup cross-platform environment for the application.
        
        Returns:
            Dictionary with setup information
        """
        setup_info = {
            "platform": self.platform_info.get_info_dict(),
            "paths": {},
            "environment": {}
        }
        
        # Setup directories
        config_dir = self.path_manager.get_config_dir()
        data_dir = self.path_manager.get_data_dir()
        cache_dir = self.path_manager.get_cache_dir()
        
        # Ensure directories exist
        for name, path in [("config", config_dir), ("data", data_dir), ("cache", cache_dir)]:
            if self.path_manager.ensure_dir(path):
                setup_info["paths"][name] = str(path)
            else:
                logger.warning(f"Failed to create {name} directory: {path}")
        
        # Set up environment variables
        env_vars = {
            "RETRIEVAL_FREE_CONFIG_DIR": str(config_dir),
            "RETRIEVAL_FREE_DATA_DIR": str(data_dir),
            "RETRIEVAL_FREE_CACHE_DIR": str(cache_dir),
        }
        
        for name, value in env_vars.items():
            self.set_env_var(name, value)
            setup_info["environment"][name] = value
        
        return setup_info


class DependencyManager:
    """Cross-platform dependency management."""
    
    def __init__(self):
        """Initialize dependency manager."""
        self.platform_info = PlatformInfo()
        self.process_manager = ProcessManager()
    
    def check_python_version(self, min_version: Tuple[int, int] = (3, 10)) -> bool:
        """Check if Python version meets requirements.
        
        Args:
            min_version: Minimum required version tuple
            
        Returns:
            True if version is sufficient
        """
        current = self.platform_info.python_version[:2]
        return current >= min_version
    
    def check_dependency(self, package_name: str) -> Dict[str, Any]:
        """Check if a dependency is available.
        
        Args:
            package_name: Name of the package to check
            
        Returns:
            Dictionary with dependency information
        """
        try:
            __import__(package_name)
            return {
                "available": True,
                "package": package_name,
                "error": None
            }
        except ImportError as e:
            return {
                "available": False,
                "package": package_name,
                "error": str(e)
            }
    
    def check_system_dependencies(self) -> Dict[str, Dict[str, Any]]:
        """Check system-level dependencies.
        
        Returns:
            Dictionary mapping dependency names to status
        """
        dependencies = {
            "git": self._check_git(),
            "curl": self._check_curl(),
        }
        
        return dependencies
    
    def _check_git(self) -> Dict[str, Any]:
        """Check git availability."""
        git_path = self.process_manager.find_executable("git")
        if git_path:
            returncode, stdout, stderr = self.process_manager.run_command(["git", "--version"])
            return {
                "available": returncode == 0,
                "path": git_path,
                "version": stdout.strip() if returncode == 0 else None,
                "error": stderr if returncode != 0 else None
            }
        else:
            return {
                "available": False,
                "path": None,
                "version": None,
                "error": "git not found in PATH"
            }
    
    def _check_curl(self) -> Dict[str, Any]:
        """Check curl availability."""
        curl_path = self.process_manager.find_executable("curl")
        if curl_path:
            returncode, stdout, stderr = self.process_manager.run_command(["curl", "--version"])
            return {
                "available": returncode == 0,
                "path": curl_path,
                "version": stdout.split('\n')[0] if returncode == 0 else None,
                "error": stderr if returncode != 0 else None
            }
        else:
            return {
                "available": False,
                "path": None,
                "version": None,
                "error": "curl not found in PATH"
            }
    
    def get_system_report(self) -> Dict[str, Any]:
        """Get comprehensive system report.
        
        Returns:
            System compatibility report
        """
        return {
            "platform": self.platform_info.get_info_dict(),
            "python_compatible": self.check_python_version(),
            "system_dependencies": self.check_system_dependencies(),
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }


# Global instances
_platform_info: Optional[PlatformInfo] = None
_path_manager: Optional[PathManager] = None
_environment_manager: Optional[EnvironmentManager] = None


def get_platform_info() -> PlatformInfo:
    """Get global platform info instance."""
    global _platform_info
    if _platform_info is None:
        _platform_info = PlatformInfo()
    return _platform_info


def get_path_manager() -> PathManager:
    """Get global path manager instance."""
    global _path_manager
    if _path_manager is None:
        _path_manager = PathManager()
    return _path_manager


def get_environment_manager() -> EnvironmentManager:
    """Get global environment manager instance."""
    global _environment_manager
    if _environment_manager is None:
        _environment_manager = EnvironmentManager()
    return _environment_manager


def setup_cross_platform_environment() -> Dict[str, Any]:
    """Setup cross-platform environment for the application.
    
    Returns:
        Setup information dictionary
    """
    env_manager = get_environment_manager()
    return env_manager.setup_environment()