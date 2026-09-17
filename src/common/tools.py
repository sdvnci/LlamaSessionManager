from hashlib import new
from pathlib import Path


def ensure_directory(directory: Path) -> bool:
    directory.mkdir(exist_ok=True, parents=True)
    return directory.is_dir()


def compute_file_hash(file_path: Path, algorithm="sha256"):
    """Compute the hash of a file using the specified algorithm."""
    hash_func = new(algorithm)

    with file_path.open("rb") as file:
        while chunk := file.read(8192):
            hash_func.update(chunk)

    return hash_func.hexdigest()
