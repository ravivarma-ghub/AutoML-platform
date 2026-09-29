"""
Data loader module for the Intelligent AutoML Platform.

Supports CSV, Parquet, Excel, JSON formats with automatic encoding detection,
delimiter inference, date parsing, and gzip decompression.
"""

from __future__ import annotations

import gzip
import io
import logging
import os
from pathlib import Path
from typing import Optional

import chardet
import pandas as pd

logger = logging.getLogger(__name__)


class DataLoaderError(Exception):
    """Raised when a data loading operation fails."""


class DataLoader:
    """
    Loads tabular data from various file formats into pandas DataFrames.

    Supported formats
    -----------------
    - CSV  (.csv, .csv.gz)
    - Parquet (.parquet, .pq)
    - Excel  (.xlsx, .xls, .xlsm)
    - JSON  (.json, .jsonl, .ndjson)

    Features
    --------
    - Automatic encoding detection (utf-8, latin-1, cp1252, …)
    - Auto-delimiter inference for CSV files
    - Transparent gzip decompression
    - Best-effort date column parsing
    - Informative error messages
    """

    # Maximum bytes used for encoding sniffing / delimiter detection
    _SNIFF_BYTES: int = 65_536

    # Candidate delimiters tried during auto-detection
    _DELIMITERS: list[str] = [",", ";", "\t", "|", " "]

    # Encodings tried in order when chardet is uncertain
    _ENCODING_FALLBACKS: list[str] = ["utf-8", "latin-1", "cp1252"]

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def load(self, file_path: str) -> pd.DataFrame:
        """
        Load a file from disk into a DataFrame.

        Parameters
        ----------
        file_path : str
            Absolute or relative path to the file.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        DataLoaderError
            If the file cannot be found, read, or parsed.
        """
        path = Path(file_path)
        if not path.exists():
            raise DataLoaderError(f"File not found: {file_path!r}")
        if not path.is_file():
            raise DataLoaderError(f"Path is not a file: {file_path!r}")

        logger.info("Loading file: %s  (%.2f MB)", path.name, path.stat().st_size / 1_048_576)

        try:
            content_bytes = path.read_bytes()
        except PermissionError as exc:
            raise DataLoaderError(f"Permission denied reading {file_path!r}: {exc}") from exc

        return self.load_from_bytes(content_bytes, path.name)

    def load_from_bytes(self, content: bytes, filename: str) -> pd.DataFrame:
        """
        Load raw bytes into a DataFrame using ``filename`` to determine format.

        Parameters
        ----------
        content : bytes
            Raw file content (may be gzip-compressed).
        filename : str
            Original filename, used for extension-based format detection.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        DataLoaderError
            If the format is unsupported or parsing fails.
        """
        if not content:
            raise DataLoaderError(f"File {filename!r} is empty.")

        # Decompress gzip if needed
        content = self._decompress_if_needed(content, filename)
        # Strip .gz suffix for subsequent extension matching
        clean_name = filename.lower().removesuffix(".gz")

        try:
            if clean_name.endswith((".csv", ".txt", ".tsv", ".dat")):
                df = self._load_csv(content, clean_name)
            elif clean_name.endswith((".parquet", ".pq")):
                df = self._load_parquet(content)
            elif clean_name.endswith((".xlsx", ".xls", ".xlsm", ".xlsb")):
                df = self._load_excel(content)
            elif clean_name.endswith((".json",)):
                df = self._load_json(content)
            elif clean_name.endswith((".jsonl", ".ndjson")):
                df = self._load_jsonlines(content)
            else:
                raise DataLoaderError(
                    f"Unsupported file format for {filename!r}. "
                    "Supported: CSV, Parquet, Excel (.xlsx/.xls), JSON, JSONL."
                )
        except DataLoaderError:
            raise
        except Exception as exc:
            raise DataLoaderError(
                f"Failed to parse {filename!r}: {exc}"
            ) from exc

        df = self._post_process(df)
        logger.info(
            "Loaded %s: shape=%s, memory=%.2f MB",
            filename,
            df.shape,
            df.memory_usage(deep=True).sum() / 1_048_576,
        )
        return df

    def infer_delimiter(self, content: str) -> str:
        """
        Auto-detect the field delimiter used in a CSV string.

        Strategy
        --------
        1. Use ``csv.Sniffer`` on a representative head of the content.
        2. If sniffer fails, count occurrences of each candidate delimiter
           across the first 20 lines and pick the most consistent one.

        Parameters
        ----------
        content : str
            Raw text content of a CSV file.

        Returns
        -------
        str
            Single-character delimiter (defaults to ``","`` if detection fails).
        """
        import csv

        head = "\n".join(content.splitlines()[:50])
        try:
            dialect = csv.Sniffer().sniff(head, delimiters="".join(self._DELIMITERS))
            logger.debug("Sniffer detected delimiter: %r", dialect.delimiter)
            return dialect.delimiter
        except csv.Error:
            pass

        # Fallback: consistency check across the first 20 data lines
        lines = [ln for ln in content.splitlines() if ln.strip()][:20]
        best_delim = ","
        best_score = -1

        for delim in self._DELIMITERS:
            counts = [line.count(delim) for line in lines]
            if not counts or max(counts) == 0:
                continue
            # Score = mean count per line, penalised by variance
            import statistics
            mean_c = statistics.mean(counts)
            variance = statistics.variance(counts) if len(counts) > 1 else 0
            score = mean_c - variance
            if score > best_score:
                best_score = score
                best_delim = delim

        logger.debug("Fallback delimiter detection chose: %r", best_delim)
        return best_delim

    def get_file_info(self, file_path: str) -> dict:
        """
        Return metadata about a file without loading its full content.

        Parameters
        ----------
        file_path : str
            Path to the file.

        Returns
        -------
        dict
            Keys: ``name``, ``extension``, ``size_bytes``, ``size_mb``,
            ``is_compressed``, ``detected_encoding``, ``format``.

        Raises
        ------
        DataLoaderError
            If the file does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise DataLoaderError(f"File not found: {file_path!r}")

        stat = path.stat()
        size_bytes = stat.st_size
        name = path.name
        suffix = path.suffix.lower()
        is_compressed = suffix == ".gz"

        clean_suffix = Path(path.stem).suffix.lower() if is_compressed else suffix

        format_map = {
            ".csv": "CSV",
            ".tsv": "CSV",
            ".txt": "CSV",
            ".dat": "CSV",
            ".parquet": "Parquet",
            ".pq": "Parquet",
            ".xlsx": "Excel",
            ".xls": "Excel",
            ".xlsm": "Excel",
            ".xlsb": "Excel",
            ".json": "JSON",
            ".jsonl": "JSON Lines",
            ".ndjson": "JSON Lines",
        }
        file_format = format_map.get(clean_suffix, "Unknown")

        # Sniff encoding from first _SNIFF_BYTES bytes
        detected_encoding = "unknown"
        try:
            raw_bytes = path.read_bytes()
            sniff_bytes = raw_bytes[: self._SNIFF_BYTES]
            if is_compressed:
                try:
                    sniff_bytes = gzip.decompress(raw_bytes)[: self._SNIFF_BYTES]
                except Exception:
                    pass
            result = chardet.detect(sniff_bytes)
            detected_encoding = result.get("encoding") or "unknown"
        except Exception:
            pass

        return {
            "name": name,
            "extension": suffix,
            "clean_extension": clean_suffix,
            "format": file_format,
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / 1_048_576, 4),
            "is_compressed": is_compressed,
            "detected_encoding": detected_encoding,
            "last_modified": stat.st_mtime,
        }

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _decompress_if_needed(self, content: bytes, filename: str) -> bytes:
        """Transparently decompress gzip content."""
        if filename.lower().endswith(".gz"):
            try:
                decompressed = gzip.decompress(content)
                logger.debug("Decompressed gzip file: %s", filename)
                return decompressed
            except gzip.BadGzipFile as exc:
                raise DataLoaderError(
                    f"File {filename!r} has .gz extension but is not valid gzip: {exc}"
                ) from exc
        return content

    def _detect_encoding(self, raw_bytes: bytes) -> str:
        """
        Detect the character encoding of raw bytes.

        Uses chardet for an initial guess, then validates with a list of
        fallbacks to avoid mojibake.
        """
        sniff = raw_bytes[: self._SNIFF_BYTES]
        result = chardet.detect(sniff)
        confidence = result.get("confidence") or 0.0
        encoding = result.get("encoding") or ""

        if confidence >= 0.85 and encoding:
            logger.debug("chardet encoding: %s (confidence=%.0f%%)", encoding, confidence * 100)
            return encoding

        # Low-confidence: try fallbacks
        for enc in self._ENCODING_FALLBACKS:
            try:
                raw_bytes.decode(enc)
                logger.debug("Encoding fallback succeeded: %s", enc)
                return enc
            except (UnicodeDecodeError, LookupError):
                continue

        return "utf-8"  # last resort

    def _load_csv(self, content: bytes, filename: str) -> pd.DataFrame:
        """Parse CSV bytes into a DataFrame."""
        encoding = self._detect_encoding(content)

        try:
            text = content.decode(encoding, errors="replace")
        except (UnicodeDecodeError, LookupError):
            text = content.decode("utf-8", errors="replace")

        delimiter = self.infer_delimiter(text)

        common_kwargs: dict = dict(
            sep=delimiter,
            encoding=encoding,
            on_bad_lines="warn",
            low_memory=False,
        )

        try:
            df = pd.read_csv(io.BytesIO(content), **common_kwargs)
        except Exception:
            # Re-try with utf-8 + error replacement
            df = pd.read_csv(
                io.StringIO(text),
                sep=delimiter,
                on_bad_lines="warn",
                low_memory=False,
            )

        if df.empty:
            raise DataLoaderError(
                f"CSV file {filename!r} produced an empty DataFrame. "
                "Check that the file contains data rows."
            )
        return df

    def _load_parquet(self, content: bytes) -> pd.DataFrame:
        """Parse Parquet bytes into a DataFrame."""
        buf = io.BytesIO(content)
        try:
            df = pd.read_parquet(buf)
        except Exception as exc:
            raise DataLoaderError(f"Failed to read Parquet data: {exc}") from exc
        return df

    def _load_excel(self, content: bytes) -> pd.DataFrame:
        """Parse Excel bytes into a DataFrame (first sheet)."""
        buf = io.BytesIO(content)
        try:
            # Load first sheet; openpyxl / xlrd selected automatically
            excel_file = pd.ExcelFile(buf)
            sheet_name = excel_file.sheet_names[0]
            if len(excel_file.sheet_names) > 1:
                logger.warning(
                    "Excel file has %d sheets; loading first sheet %r only.",
                    len(excel_file.sheet_names),
                    sheet_name,
                )
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
        except Exception as exc:
            raise DataLoaderError(f"Failed to read Excel data: {exc}") from exc
        return df

    def _load_json(self, content: bytes) -> pd.DataFrame:
        """Parse JSON bytes into a DataFrame."""
        encoding = self._detect_encoding(content)
        text = content.decode(encoding, errors="replace")
        try:
            df = pd.read_json(io.StringIO(text))
        except ValueError:
            # Try orient="records"
            try:
                df = pd.read_json(io.StringIO(text), orient="records")
            except ValueError as exc:
                raise DataLoaderError(f"Failed to parse JSON: {exc}") from exc
        return df

    def _load_jsonlines(self, content: bytes) -> pd.DataFrame:
        """Parse JSON Lines (one JSON object per line) into a DataFrame."""
        encoding = self._detect_encoding(content)
        text = content.decode(encoding, errors="replace")
        try:
            df = pd.read_json(io.StringIO(text), lines=True)
        except ValueError as exc:
            raise DataLoaderError(f"Failed to parse JSON Lines: {exc}") from exc
        return df

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply post-load cleaning:
        - Strip leading/trailing whitespace from column names.
        - Attempt to parse object columns that look like dates.
        - Clean up unnamed index columns produced by CSV export.
        """
        # Normalise column names
        df.columns = [str(c).strip() for c in df.columns]

        # Drop spurious unnamed index columns (e.g. "Unnamed: 0")
        unnamed = [c for c in df.columns if c.startswith("Unnamed:")]
        if unnamed:
            logger.debug("Dropping unnamed index columns: %s", unnamed)
            df = df.drop(columns=unnamed)

        # Best-effort date parsing on string columns
        df = self._parse_date_columns(df)

        return df

    def _parse_date_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attempt to infer and parse datetime columns from object-type columns.

        Heuristic: column name contains a date-related keyword or the first
        non-null value can be parsed as a date.
        """
        date_keywords = {
            "date", "time", "timestamp", "created", "updated",
            "modified", "year", "month", "day", "dt",
        }
        object_cols = df.select_dtypes(include=["object"]).columns.tolist()

        for col in object_cols:
            col_lower = col.lower()
            name_hint = any(kw in col_lower for kw in date_keywords)

            # Sample non-null values to probe
            sample = df[col].dropna().head(5)
            if sample.empty:
                continue

            if name_hint:
                try:
                    df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                    null_ratio = df[col].isna().mean()
                    if null_ratio > 0.5:
                        # Conversion mostly failed — revert with object dtype
                        df[col] = df[col].astype("object")
                    else:
                        logger.debug("Parsed column %r as datetime.", col)
                except Exception:
                    pass

        return df
