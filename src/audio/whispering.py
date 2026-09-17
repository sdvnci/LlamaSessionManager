import os
from datetime import UTC, datetime
from os import PathLike
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO, Callable, Final, Iterator

# Must be set before any CUDA context is created.  The segmented allocator
# avoids the fragmentation that OOMs long-running transcribe workers.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
from numpy import ceil, float32, uint32, zeros
from numpy.typing import NDArray
from torch import cuda, inference_mode
from whisper import (  # type: ignore[import-untyped]
    DecodingOptions,
    Whisper,
    load_audio,
    load_model,
    log_mel_spectrogram,
)
from whisper import (  # type: ignore[import-untyped]
    decode as decode_whisper,
)
from whisper.audio import SAMPLE_RATE  # type: ignore[import-untyped]

from ._types import TranscriptProgress, TranscriptSegment
from .vad import VoiceActivityDetector

AudioSourceType = (
    str
    | PathLike[str]
    | NDArray[np.float32]
    | BinaryIO
    | bytes
    | bytearray
    | memoryview
)


def select_cuda_device() -> str:
    """Return the CUDA device with the most free VRAM, or ``"cpu"``.

    The Ollama container (and anything else sharing the box) may already
    occupy one card; picking the least-loaded GPU avoids OOM collisions.
    """
    if not cuda.is_available():
        return "cpu"
    if cuda.device_count() <= 1:
        return "cuda"
    best = max(
        range(cuda.device_count()),
        key=lambda i: cuda.mem_get_info(i)[0],
    )
    return f"cuda:{best}"


def default_transcriber_initializer(
    service: "TranscriptionService", model_name: str = "base"
) -> Whisper | None:
    """Load the Whisper model, reusing an already-loaded one.

    Whisper's ``load_model`` maps the fp32 checkpoint straight onto the GPU
    and then moves the fp32 model there too — a ~12 GB peak for large-v3
    that no 8 GB card survives.  Load on CPU instead and move once, keeping
    the GPU peak at the model's fp32 size (~6.2 GB).

    The weights must stay fp32: whisper's ``LayerNorm`` casts inputs to
    fp32 internally, so halving the model breaks decode.  Whisper already
    uses fp16 activations on CUDA via ``DecodingOptions(fp16=True)``.
    """
    if service.TRANSCRIBER is not None:
        return service.TRANSCRIBER

    model = load_model(model_name, device="cpu", in_memory=False)
    if service.DEVICE.startswith("cuda"):
        model = model.to(service.DEVICE)
    return model


class TranscriptionService:
    """A service that handles transcription of audio data using Whisper.

    All audio — whether from a file path, raw bytes, or a readable stream —
    is decoded through whisper's own ``load_audio`` (which invokes ffmpeg
    internally) so the mel-spectrogram stage always receives the exact format
    the model expects.
    """

    __slots__ = (
        "transcriber_init_hook",
        "_TRANSCRIBER",
        "_DEVICE",
        "_CHUNK_LIM",
        "_VAD",
    )

    def __init__(
        self,
        model_name: str = "base",
        transcriber_initializer: Callable[
            ["TranscriptionService", str], Whisper | None
        ] = default_transcriber_initializer,
        chunk_limit: uint32 = uint32(480000),
        vad_mode: int = 2,
        vad_min_speech_ratio: float = 0.0,
    ) -> None:
        self._TRANSCRIBER: Whisper | None = None
        self._DEVICE: Final[str] = select_cuda_device()
        self._CHUNK_LIM: uint32 = chunk_limit
        self._VAD: VoiceActivityDetector = VoiceActivityDetector(
            mode=vad_mode,
            min_speech_ratio=vad_min_speech_ratio,
        )
        self.transcriber_init_hook = transcriber_initializer
        self._TRANSCRIBER = self.transcriber_init_hook(self, model_name)

    @property
    def TRANSCRIBER(self) -> Whisper | None:
        """The loaded Whisper model, or ``None`` if not initialized."""
        return self._TRANSCRIBER

    @property
    def DEVICE(self) -> str:
        """Inference device: the least-loaded CUDA GPU, or ``"cpu"``."""
        return self._DEVICE

    def release_cuda_cache(self) -> None:
        """Return cached CUDA blocks to the driver (call between jobs)."""
        if self._DEVICE.startswith("cuda"):
            cuda.empty_cache()

    @property
    def VAD(self) -> VoiceActivityDetector:
        """The voice activity detector screening buffers before transcription."""
        return self._VAD

    @staticmethod
    def load_audio_file(audio_file: str | PathLike[str]) -> NDArray[np.float32]:
        """Load a compressed audio file via whisper's built-in ffmpeg decoder."""
        try:
            return load_audio(str(audio_file))
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Whisper needs the ffmpeg executable available on PATH "
                "to decode audio files. Install ffmpeg, then retry."
            ) from exc

    def _load_bytes_via_tempfile(
        self, data: bytes | bytearray | memoryview
    ) -> NDArray[np.float32]:
        """Write raw audio bytes to a temp file so whisper's ``load_audio`` handles
        decoding.  This avoids the tensor-shape mismatch that can occur when
        a custom ffmpeg pipeline produces PCM that differs subtly from what
        the Whisper mel-filterbank expects."""
        suffix = ".mp3"  # ffmpeg probes the content; extension is a hint
        with NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(bytes(data))
            tmp.flush()
            tmp_path = Path(tmp.name)
        try:
            return self.load_audio_file(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    @staticmethod
    def pad_or_trim(
        array: NDArray[np.float32],
        length: uint32 = uint32(480000),
    ) -> NDArray[np.float32]:
        """Pad with zeros or trim a 1D waveform to exactly ``length`` samples."""
        if array.shape[0] > length:
            return array[:length]
        if array.shape[0] < length:
            result = zeros(length, dtype=array.dtype)
            result[: array.shape[0]] = array
            return result
        return array

    def chunk_audio(
        self,
        audio: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Split a 1D waveform into ``(n, CHUNK_LIM)`` chunks."""
        audio_length = audio.shape[0]
        num_chunks = max(1, int(ceil(audio_length / int(self._CHUNK_LIM))))
        audios = zeros((num_chunks, int(self._CHUNK_LIM)), dtype=float32)

        if num_chunks == 1:
            audios[0] = self.pad_or_trim(audio, self._CHUNK_LIM)
        else:
            for i in range(num_chunks):
                start = i * int(self._CHUNK_LIM)
                end = min((i + 1) * int(self._CHUNK_LIM), audio_length)
                chunk = audio[start:end]
                if chunk.shape[0] < int(self._CHUNK_LIM):
                    chunk = self.pad_or_trim(chunk, self._CHUNK_LIM)
                audios[i] = chunk
        return audios

    def transcribe_audio(
        self,
        audio: NDArray[np.float32],
        batch_size: int = 8,
    ) -> str:
        """Transcribe a 1D waveform or 2D chunk batch with Whisper EN.

        Audio must be 16 kHz mono float32.
        """
        if self._TRANSCRIBER is None:
            raise RuntimeError("Failed to initialize Whisper transcriber")

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        if audio.ndim == 1:
            prepared = self.chunk_audio(audio.astype(float32, copy=False))
        elif audio.ndim == 2:
            prepared = audio.astype(float32, copy=False)
        else:
            raise ValueError("audio must be 1D or 2D")

        # Some Whisper model variants (e.g. large-v3) use 128 mel bins
        # instead of the classic 80.  Read the value from the model
        # config so the mel spectrogram always matches the conv layer.
        n_mels: int = getattr(getattr(self._TRANSCRIBER, "dims", None), "n_mels", 80)

        device = self._TRANSCRIBER.device
        options = DecodingOptions(
            temperature=0,
            fp16=device.type == "cuda",
            language="en",
            without_timestamps=True,
            beam_size=1,
        )

        results: list[str] = []
        with inference_mode():
            for i in range(0, prepared.shape[0], batch_size):
                batch = prepared[i : i + batch_size]
                mel = log_mel_spectrogram(batch, n_mels=n_mels, device=device)
                decoded = decode_whisper(self._TRANSCRIBER, mel, options)
                decoded_results = decoded if isinstance(decoded, list) else [decoded]
                for r in decoded_results:
                    if r.text:
                        results.append(r.text.strip())

        return " ".join(results)

    def transcribe_file(
        self,
        audio_file: (
            str
            | PathLike[str]
            | NDArray[np.float32]
            | BinaryIO
            | bytes
            | bytearray
            | memoryview
        ),
        batch_size: int = 8,
    ) -> str:
        """Transcribe from a file path, numpy waveform, raw bytes, or stream."""
        if isinstance(audio_file, np.ndarray):
            return self.transcribe_audio(audio_file, batch_size=batch_size)

        if isinstance(audio_file, (str, PathLike)):
            return self.transcribe_audio(
                self.load_audio_file(audio_file),
                batch_size=batch_size,
            )

        if isinstance(audio_file, (bytes, bytearray, memoryview)):
            return self.transcribe_audio(
                self._load_bytes_via_tempfile(audio_file),
                batch_size=batch_size,
            )

        if hasattr(audio_file, "read"):
            raw = audio_file.read()
            return self.transcribe_audio(
                self._load_bytes_via_tempfile(raw),
                batch_size=batch_size,
            )

        raise TypeError(
            "audio_file must be a path, numpy waveform, bytes-like object, "
            "or readable binary stream"
        )

    def _load_waveform(
        self,
        source: "AudioSourceType",
    ) -> NDArray[np.float32]:
        """Decode any supported audio source into a single 1D float32 waveform.

        Bytes and streams are decoded through whisper's ``load_audio``
        (via a temp file) to guarantee mel-spectrogram compatibility.
        """
        if isinstance(source, np.ndarray):
            return source.astype(float32, copy=False)

        if isinstance(source, (str, PathLike)):
            return self.load_audio_file(source)

        if isinstance(source, (bytes, bytearray, memoryview)):
            return self._load_bytes_via_tempfile(source)

        if hasattr(source, "read"):
            return self._load_bytes_via_tempfile(source.read())

        raise TypeError(
            "source must be a path, numpy waveform, bytes-like object, "
            "or readable binary stream"
        )

    def _iter_source_chunks(
        self,
        source: "AudioSourceType",
    ) -> Iterator[NDArray[np.float32]]:
        """Yield 1D waveform chunks from any supported audio source."""
        yield from self.chunk_audio(self._load_waveform(source))

    def iter_transcribed_progress(
        self,
        source: "AudioSourceType",
        transcript_id: str,
        enable_vad: bool = True,
    ) -> Iterator[TranscriptProgress]:
        """Transcribe ``source`` chunk-by-chunk, yielding ``TranscriptProgress``.

        Exactly one progress record is yielded per source chunk so callers can
        render a live percentage.  ``fraction`` is the portion of the recording
        processed so far (``0.0``–``1.0``); chunks that produce no text —
        VAD-filtered or silent — still advance ``fraction`` and come back with
        ``segment=None``.
        """
        samples_per_chunk = int(self._CHUNK_LIM)
        seconds_per_chunk = samples_per_chunk / SAMPLE_RATE
        waveform = self._load_waveform(source)
        duration_seconds = float(waveform.shape[0] / SAMPLE_RATE)
        for index, chunk in enumerate(self.chunk_audio(waveform)):
            if chunk.shape[0] != samples_per_chunk:
                chunk = self.pad_or_trim(chunk, self._CHUNK_LIM)
            end = min((index + 1) * seconds_per_chunk, duration_seconds)
            segment: TranscriptSegment | None = None
            if not enable_vad or self._VAD.contains_speech(chunk):
                text = self.transcribe_audio(chunk, batch_size=1)
                if text:
                    segment = TranscriptSegment(
                        uid=transcript_id,
                        sequence=index,
                        content=text,
                        start=index * seconds_per_chunk,
                        end=end,
                        created_at=datetime.now(UTC),
                    )
            yield TranscriptProgress(
                segment=segment,
                fraction=(end / duration_seconds) if duration_seconds > 0 else 1.0,
                processed_seconds=end,
                duration_seconds=duration_seconds,
            )

    def iter_transcribed_segments(
        self,
        source: "AudioSourceType",
        transcript_id: str,
        enable_vad: bool = True,
    ) -> Iterator[TranscriptSegment]:
        """Transcribe ``source`` chunk-by-chunk, yielding text segments only.

        Timestamps are estimated from the fixed chunk length.  When
        ``enable_vad`` is set, buffers without detectable speech are skipped
        entirely — they waste GPU time and can hallucinate from ambient noise.
        For per-chunk progress reporting use ``iter_transcribed_progress``.
        """
        for progress in self.iter_transcribed_progress(
            source, transcript_id, enable_vad=enable_vad
        ):
            if progress.segment is not None:
                yield progress.segment
