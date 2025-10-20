import importlib.util
import sys


class LazyImportFinder:
    def __init__(self):
        self._importing = set()  # Track what we're currently importing

    def find_spec(self, fullname, path, target=None):
        # Prevent recursion
        if fullname in self._importing:
            return None

        if not self._should_be_lazy(fullname):
            return None

        self._importing.add(fullname)
        try:
            # Find the real spec using the original meta_path (excluding ourselves)
            spec = None
            for finder in sys.meta_path:
                if finder is self:
                    continue
                if hasattr(finder, 'find_spec'):
                    spec = finder.find_spec(fullname, path, target)
                    if spec is not None:
                        break

            if spec is not None and hasattr(spec.loader, 'exec_module'):
                # Wrap with LazyLoader
                spec.loader = importlib.util.LazyLoader(spec.loader)

            return spec
        finally:
            self._importing.discard(fullname)

    def _should_be_lazy(self, fullname):
        for lazy_module in _lazy_modules:
            if fullname == lazy_module or fullname.startswith(lazy_module + '.'):
                return True
        return False


# Mark which modules should be lazy (base modules)
# would love to lazy load torch and transformers but their import process is too complicated
_lazy_modules = {'bsparse', 'faiss'}
sys.meta_path.insert(0, LazyImportFinder())
