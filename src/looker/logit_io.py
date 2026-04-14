"""Responsible for loading and saving model outputs."""

import contextlib
import json
import os
import pathlib
import re
from typing import Generator, Self

import torch


class LogitWriter:
    """The `LogitIO` class is responsible for (de)serialization of output
    logits.
    """

    def __init__(self: Self, base_dir: pathlib.Path):
        self._base_dir = base_dir
        self._written_files: list[pathlib.Path] = []
        self._created_dir = False

        if not os.path.exists(self._base_dir):
            os.mkdir(self._base_dir)
            self._created_dir = True

    def write_logits(
        self: Self, chunk_no: int, model: str, logits: torch.Tensor
    ) -> None:
        """Write logits for a specific chunk, model pair.

        Args:
            chunk_no: Chunk number.
            model: Model name.
            logits: Output logits for a specific chunk.
        """
        logit_fname = self._base_dir / f"logits_{model}_{chunk_no}.pt"

        with open(logit_fname, "wb", encoding="utf-8") as logit_file:
            torch.save(logits, logit_file)

    def write_prompts(
        self: Self, chunk_no: int, indices: list[int], prompts: torch.Tensor
    ) -> None:
        """Write prompts for a specific chunk.

        Args:
            chunk_no: Chunk number.
            indices: Dataset indices of prompts.
            prompts: Prompts in the chunk.
        """
        prompt_fname = self._base_dir / f"prompts_{chunk_no}.pt"

        with open(prompt_fname, "wb") as prompt_file:
            torch.save({"indices": indices, "prompts": prompts}, prompt_file)

    def finalize(self: Self) -> None:
        """Compile metadata on which models were written, which chunks are
        available. Write all results to a manifest.json file.
        """
        manifest_path = self._base_dir / "manifest.json"

        metadata = {
            "models": self._get_unique_models(),
            "chunks": self._get_total_chunks(),
        }

        with open(manifest_path, "w", encoding="utf-8") as manifest:
            json.dump(metadata, manifest, indent=2)

    def clear(self: Self) -> None:
        """Clean up all files written. Called in the event of an error. If the
        base directory was created in conjunction, also delete that.
        """
        for file in self._written_files:
            os.remove(file)

        if self._created_dir:
            os.rmdir(self._base_dir)

    def _get_total_chunks(self: Self) -> int:
        def _parse_chunk(filename: pathlib.Path) -> int:
            matches = re.search(r"logits_.+_(\d+)\.pt", str(filename))
            assert matches is not None  # Can never be None.
            return int(matches.group(1))

        return max(_parse_chunk(filename) for filename in self._written_files)

    def _get_unique_models(self: Self) -> list[str]:
        def _parse_model_name(filename: pathlib.Path) -> str:
            matches = re.search(r"logits_(.+)_\d+\.pt", str(filename))
            assert matches is not None  # Can never be None.
            return matches.group(1)

        # Sort so that output is deterministic.
        return sorted(
            set(_parse_model_name(filename) for filename in self._written_files)
        )


@contextlib.contextmanager
def logit_writer(base_dir: pathlib.Path) -> Generator[LogitWriter, None, None]:
    """Open a `LogitWriter` instance.

    Args:
        base_dir: Directory in which to write logits.
    """
    writer = LogitWriter(base_dir)

    try:
        yield writer

    except:  # pylint: disable=bare-except
        writer.clear()

    else:
        writer.finalize()


def available_models(base_dir: pathlib.Path) -> list[str]:
    """Get a list of all available models.

    Args:
        base_dir: Directory in which logits were saved.

    Returns:
        A list of all models whose logits were written in `base_dir`.
    """
    with open(base_dir / "manifest.json", encoding="utf-8") as manifest:
        metadata = json.load(manifest)

        return metadata["models"]


def read_logits(
    base_dir: pathlib.Path, models: list[str], prompts: list[str] | None = None
) -> dict[str, dict[str, torch.Tensor]]:
    """Read logits and prompts.

    Args:
        models: List of model names to retrieve prompts and their associated
            outputs logits for.
        prompts: If specified, a list of prompts to filter outputs to. Otherwise
            all logits are returned. Defaults to `None`.

    Returns:
        A dictionary of `{prompt: {model_name: logits}}`.
    """
    del base_dir, models, prompts

    return {}
