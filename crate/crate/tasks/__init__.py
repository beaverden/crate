"""Slow libcrate work run off the UI thread: scanning the library, enriching the catalog, and downloading
models."""

from crate.tasks.download_models import DownloadModelsTask
from crate.tasks.enrich import EnrichTask
from crate.tasks.scan import ScanTask

__all__ = ["DownloadModelsTask", "EnrichTask", "ScanTask"]
