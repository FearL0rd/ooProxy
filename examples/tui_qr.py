#!/usr/bin/env python3
"""
Companion for .ooProxy/tools/tui_qr.json
Generates and prints an ASCII/TUI QR code to the terminal.
"""
import argparse
import sys

import qrcode


def _is_dark(cell: bool, invert: bool) -> bool:
    return not cell if invert else cell


def _render_block_lines(matrix: list[list[bool]], invert: bool) -> list[str]:
    return [
        "".join("██" if _is_dark(cell, invert) else "  " for cell in row)
        for row in matrix
    ]


def _render_half_block_lines(matrix: list[list[bool]], invert: bool) -> list[str]:
    lines: list[str] = []
    width = len(matrix[0]) if matrix else 0

    for row_index in range(0, len(matrix), 2):
        top_row = matrix[row_index]
        bottom_row = matrix[row_index + 1] if row_index + 1 < len(matrix) else [False] * width
        chars: list[str] = []

        for top_cell, bottom_cell in zip(top_row, bottom_row):
            top_dark = _is_dark(top_cell, invert)
            bottom_dark = _is_dark(bottom_cell, invert)
            if top_dark and bottom_dark:
                chars.append("█")
            elif top_dark:
                chars.append("▀")
            elif bottom_dark:
                chars.append("▄")
            else:
                chars.append(" ")

        lines.append("".join(chars))

    return lines


def _render_lines(matrix: list[list[bool]], invert: bool, style: str) -> list[str]:
    if style == "half":
        return _render_half_block_lines(matrix, invert)
    return _render_block_lines(matrix, invert)


def main() -> None:
    parser = argparse.ArgumentParser(description="Display TUI QR code in terminal")
    parser.add_argument("--text", required=True, help="Text or URL to encode")
    parser.add_argument("--border", type=int, default=1, help="Border size in modules")
    parser.add_argument("--invert", action="store_true", help="Swap dark/light")
    parser.add_argument(
        "--style",
        choices=("block", "half"),
        default="half",
        help="Render style: 'block' for the original wide blocks, 'half' for a more compact QR",
    )
    args = parser.parse_args()

    if not args.text.strip():
        parser.error("Text cannot be empty")

    # Build QR
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=1,
        border=args.border,
    )
    qr.add_data(args.text)
    qr.make(fit=True)

    # Render to terminal glyphs.
    matrix = qr.get_matrix()
    sys.stdout.write("\nQRCode for "+args.text+"\n")
    lines = _render_lines(matrix, invert=args.invert, style=args.style)
    sys.stdout.write("\n".join(lines))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
