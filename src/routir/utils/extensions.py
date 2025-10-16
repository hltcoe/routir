import importlib.metadata
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from . import logger


def _explore_modules(module: ModuleType, scope=None):
    if scope is None:
        scope = globals()

    names = getattr(module, "__all__", [n for n in dir(module) if not n.startswith("_")])
    for name in names:
        scope[name] = getattr(module, name)


def load_all_extensions(user_specified_files=None, scope=None, use_entry_points=True):
    """
    Load routir extensions using both naming convention and entry points.

    Args:
        user_specified_files: List of files to import
        scope: Namespace to import into (defaults to globals())
        use_entry_points: If True, prefer entry points; if False, use prefix only
    """
    if scope is None:
        scope = globals()

    if user_specified_files is None:
        user_specified_files = []

    loaded = []

    for fn in user_specified_files:
        try:
            fn = Path(fn).resolve()

            if not fn.is_file():
                raise FileNotFoundError(f"{fn} is not a valid file")

            logger.warning(f"Importing from `{fn}`: should only do so if you trust the source. ")
            spec = importlib.util.spec_from_file_location(fn.stem, str(fn))
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            _explore_modules(module, scope)

            loaded.append(f"{fn.stem} (script)")

        except Exception as e:
            logger.warning(f"Failed to load from file {fn}: {e}")

    if use_entry_points:
        # Try entry points first
        entry_points = importlib.metadata.entry_points()

        if hasattr(entry_points, "select"):
            routir_entries = entry_points.select(group="routir.extensions")
        else:
            routir_entries = entry_points.get("routir.extensions", [])

        for entry_point in routir_entries:
            try:
                plugin = entry_point.load()
                _explore_modules(plugin, scope)
                loaded.append(f"{entry_point.name} (entry point)")
            except Exception as e:
                logger.warning(f"Failed to load {entry_point.name}: {e}")

    # Also scan for routir_* packages
    for dist in importlib.metadata.distributions():
        package_name = dist.metadata["Name"]

        if package_name.startswith("routir_"):
            try:
                module_name = package_name.replace("-", "_")

                # Skip if already loaded via entry point or files
                if any(module_name in item for item in loaded):
                    continue

                module = importlib.import_module(module_name)
                _explore_modules(module, scope)

                loaded.append(f"{package_name} (prefix)")
            except Exception as e:
                logger.warning(f"Failed to load {package_name}: {e}")

    if len(loaded) > 0:
        logger.warning("Loaded the following extensions: " + "\n    ".join([""] + loaded))
