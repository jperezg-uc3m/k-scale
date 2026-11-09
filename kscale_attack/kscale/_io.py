"""Persistence shared by the independent pipeline stages."""
import json
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from numbers import Integral


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def read_metadata(directory):
    return json.loads((Path(directory) / "metadata.json").read_text())


def write_metadata(directory, **metadata):
    (Path(directory) / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )


def require_dataset(Sigma, metadata):
    if metadata["dataset_id"] != read_metadata(Sigma)["dataset_id"]:
        raise ValueError("Stages refer to different trajectory datasets.")


@contextmanager
def new_stage(output):
    """Publish a complete stage; never overwrite an existing result."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        yield temporary
        if output.exists():
            raise FileExistsError(output)
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
