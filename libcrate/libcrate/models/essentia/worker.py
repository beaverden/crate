"""The child process that holds the Essentia models for EssentiaClient (see models.workers).

    python -m libcrate.models.essentia.worker SOCKET_FD MODELS_DIR MAX_SECONDS
"""

import sys
from multiprocessing.connection import Connection
from pathlib import Path

from libcrate.models.essentia.models import EssentiaError, Models
from libcrate.models.workers import serve

if __name__ == "__main__":
    serve(Connection(int(sys.argv[1])), lambda: Models(Path(sys.argv[2]), float(sys.argv[3])), Models.REQUESTS,
          EssentiaError, "Essentia")
