"""
Append DuckDB's extension metadata footer to a shared library.

DuckDB refuses to ``LOAD`` a file without this footer.  It is 534 bytes: a
WebAssembly custom-section header (so Wasm builds stay valid), eight
32-byte NUL-padded ASCII fields stored last to first, and 256 bytes reserved
for a signature, left empty (the extension is unsigned).

Written from the format of ``scripts/append_extension_metadata.py`` in
duckdb/extension-ci-tools at commit 7ab636a59975d063a324b1b3b97ffaf39304fe8c,
which carries no licence and so is not copied here.

Usage::

    python append_metadata.py LIBRARY OUTPUT --name batchcorder
        --platform linux_amd64 --duckdb-version v1.2.0 --extension-version 0.1.0
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


FIELD_SIZE = 32
SIGNATURE_SIZE = 256

# Custom section id 0, payload length 531 as LEB128 (0x93 0x04), a 16-byte
# section name, then the field block's length, 512, as LEB128 (0x80 0x04).
SECTION_HEADER = bytes([0, 0x93, 0x04, 16]) + b"duckdb_signature" + bytes([0x80, 0x04])


def field(value: str) -> bytes:
    """
    Encode `value` as a NUL-padded 32-byte ASCII field.

    Returns
    -------
    bytes
        The 32-byte field.

    Raises
    ------
    ValueError
        If `value` is longer than 32 bytes.

    """
    encoded = value.encode("ascii")
    if len(encoded) > FIELD_SIZE:
        raise ValueError(f"metadata field longer than {FIELD_SIZE} bytes: {value!r}")
    return encoded.ljust(FIELD_SIZE, b"\0")


def footer(*, platform: str, duckdb_version: str, extension_version: str) -> bytes:
    """
    Build the footer for the C_STRUCT ABI.

    For that ABI, `duckdb_version` is the minimum C API version the extension
    was built against.

    Returns
    -------
    bytes
        The 534-byte footer.

    """
    fields = [
        "",  # FIELD8, unused
        "",  # FIELD7, unused
        "",  # FIELD6, unused
        "C_STRUCT",  # FIELD5, ABI type
        extension_version,  # FIELD4
        duckdb_version,  # FIELD3
        platform,  # FIELD2
        "4",  # FIELD1, marks a DuckDB extension
    ]
    return SECTION_HEADER + b"".join(map(field, fields)) + b"\0" * SIGNATURE_SIZE


def main() -> None:
    """Append the footer to a copy of the library, as the command line says."""
    parser = argparse.ArgumentParser(
        description="Append DuckDB's extension metadata footer to a shared library."
    )
    parser.add_argument("library", type=Path, help="the built shared library")
    parser.add_argument("output", type=Path, help="the .duckdb_extension to write")
    parser.add_argument(
        "--name",
        required=True,
        help="extension name (unused in the footer; checked against the output name)",
    )
    parser.add_argument(
        "--platform",
        required=True,
        help="DuckDB platform, e.g. linux_amd64 (PRAGMA platform)",
    )
    parser.add_argument(
        "--duckdb-version", required=True, help="minimum C API version, e.g. v1.2.0"
    )
    parser.add_argument("--extension-version", required=True)
    args = parser.parse_args()

    if args.output.name != f"{args.name}.duckdb_extension":
        parser.error(
            f"output must be named {args.name}.duckdb_extension: DuckDB derives the entrypoint from it"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".tmp")
    shutil.copyfile(args.library, tmp)
    with tmp.open("ab") as out:
        out.write(
            footer(
                platform=args.platform,
                duckdb_version=args.duckdb_version,
                extension_version=args.extension_version,
            )
        )
    tmp.replace(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
