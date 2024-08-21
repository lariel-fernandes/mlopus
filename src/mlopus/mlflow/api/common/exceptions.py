from pathlib import Path


class FailedToPublishArtifact(Exception):
    """Failed to add artifact."""

    def __init__(self, source: Path):
        super().__init__(str(source))
        self.source = source
