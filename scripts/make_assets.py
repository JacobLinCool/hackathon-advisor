#!/usr/bin/env python3
from __future__ import annotations

import random
import struct
import zlib
from pathlib import Path


def main() -> None:
    out = Path("static/assets/parchment.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_png(out, 1280, 760)
    print(f"wrote {out}")


def write_png(path: Path, width: int, height: int) -> None:
    random.seed(47)
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            dx = abs(x / width - 0.5)
            dy = abs(y / height - 0.5)
            vignette = int((dx + dy) * 58)
            grain = random.randint(-12, 12)
            wave = int(8 * random.random() + 6 * ((x * y) % 17 == 0))
            r = clamp(217 - vignette + grain + wave)
            g = clamp(190 - vignette + grain)
            b = clamp(137 - vignette + grain // 2)
            row.extend([r, g, b, 255])
        rows.append(bytes(row))
    raw = b"".join(rows)
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def clamp(value: int) -> int:
    return max(0, min(255, value))


if __name__ == "__main__":
    main()
