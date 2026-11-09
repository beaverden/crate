"""The child process that holds a CLAP model for ClapClient (see models.workers).

    python -m libcrate.models.clap.worker SOCKET_FD MODEL MODEL_FOLDER THREADS
"""

import sys
from multiprocessing.connection import Connection
from pathlib import Path

from libcrate.models.clap.models import ClapError, Models
from libcrate.models.workers import serve

if __name__ == "__main__":
    serve(Connection(int(sys.argv[1])), lambda: Models(sys.argv[2], Path(sys.argv[3]), int(sys.argv[4])), Models.REQUESTS,
          ClapError, "CLAP")
