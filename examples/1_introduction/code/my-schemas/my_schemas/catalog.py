import mlopus

from . import foobar, foobar_no_dump


class MyCatalog(mlopus.artschema.ArtifactsCatalog):
    """Mapping of fields to artifact type."""

    foobar: foobar.Artifact
    foobar_no_dump: foobar_no_dump.Artifact
