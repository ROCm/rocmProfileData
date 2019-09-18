import hipMarker
import uuid

class hipScopedMarker:
    def __init__(self, description):
        self.description = description
        self.uuid = uuid.uuid4()

    def __enter__(self):
        hipMarker.emitMarker(f"{self.uuid};start;{self.description}")
    def __exit__(self, etype, evalue, etraceback):
        hipMarker.emitMarker(f"{self.uuid};stop;{self.description}")
    def __del__(self):
        pass

