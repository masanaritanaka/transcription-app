import ctypes
import gc
import logging
import os

logger = logging.getLogger(__name__)


def get_rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except FileNotFoundError:
        pass
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if os.uname().sysname == "Darwin":
            return ru / (1024 * 1024)
        return ru / 1024
    except Exception:
        return 0.0


def log_memory(label: str):
    logger.info(f"[memory] {label}: RSS={get_rss_mb():.0f} MB")


def release_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass
