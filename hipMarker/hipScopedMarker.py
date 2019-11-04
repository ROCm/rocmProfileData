import roctxMarker

class hipScopedMarker:
    def __init__(self, description):
        self.description = description

    @staticmethod
    def emitMarker(description):
        roctxMarker.emitMarker(description)

    def __enter__(self):
        roctxMarker.pushMarker(f"{self.description}")
    def __exit__(self, etype, evalue, etraceback):
        roctxMarker.popMarker()
    def __del__(self):
        pass
