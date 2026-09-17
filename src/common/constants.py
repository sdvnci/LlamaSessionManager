from pathlib import Path

# Project root (…/src/common/constants.py → up three levels).
# Keeps artifacts/exports/logs on the host-mounted volumes in Docker.
CWD = Path(__file__).parent.parent.parent.resolve()

LOGDIR = CWD / "logs"
EXPORTDIR = CWD / "exports"

LOGDIR.mkdir(exist_ok=True)
EXPORTDIR.mkdir(exist_ok=True)

ARTIFACTDIR = CWD / "artifacts"
ARTIFACTDIR.mkdir(exist_ok=True)
