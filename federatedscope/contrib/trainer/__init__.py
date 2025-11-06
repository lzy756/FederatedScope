import importlib
import glob
import logging
from os.path import dirname, basename, isfile, join

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [
    basename(f)[:-3] for f in modules
    if isfile(f) and not f.endswith('__init__.py')
]

logger = logging.getLogger(__name__)

for _name in __all__:
    try:
        module = importlib.import_module(f"{__name__}.{_name}")
        globals()[_name] = module
    except Exception as error:  # pragma: no cover
        # Keep lazy loading behavior for modules whose dependencies
        # are unavailable (e.g., torch). The corresponding register_*
        # hooks simply will not be executed.
        logger.warning(
            'Skip importing federatedscope.contrib.trainer.%s due to %s',
            _name, error)
