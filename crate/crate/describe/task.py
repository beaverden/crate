from libcrate.models.clap import ClapClient, ClapError
from PySide6.QtCore import QThread, Signal

__all__ = ["DescribeTask"]


class DescribeTask(QThread):
    """A description's vector from each client's model (a CLAP model, a text model), in a background thread: a
    client's model process loads on its first use (several seconds), then each description takes a fraction of a
    second.

    finished_ok(vectors), one per client in order, or failed(message) at the end.
    """

    finished_ok = Signal(list)
    failed = Signal(str)

    def __init__(self, clients: list[ClapClient], text: str, parent=None):
        super().__init__(parent)
        self.clients, self.text = clients, text

    def run(self) -> None:
        try:
            vectors = [client.query_embeddings([self.text])[0] for client in self.clients]
        except ClapError as e:
            self.failed.emit(str(e))
            return
        self.finished_ok.emit(vectors)
