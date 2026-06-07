from __future__ import annotations

from io import BytesIO
import math
import re
from typing import Any

from PIL import Image, ImageDraw, ImageFont


CANVAS_WIDTH = 1200
CANVAS_HEIGHT = 675

INK = (37, 22, 14)
INK_SOFT = (107, 78, 53)
PARCHMENT_LIGHT = (234, 215, 167)
PARCHMENT_MID = (212, 180, 118)
PARCHMENT_DARK = (185, 138, 76)
OXBLOOD = (141, 45, 38)
LEAF = (47, 122, 73)
GOLD = (182, 138, 18)
CREAM = (255, 240, 181)


def artifact_png_filename(artifact: dict[str, Any]) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(artifact.get("title") or "unwritten-page").lower())
    slug = slug.strip("-")[:60] or "unwritten-page"
    return f"{slug}.png"


def render_artifact_png(artifact: dict[str, Any]) -> bytes:
    image = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), PARCHMENT_LIGHT)
    draw = ImageDraw.Draw(image)
    _draw_parchment(draw)

    seal = artifact.get("seal") if isinstance(artifact.get("seal"), dict) else {}
    verdict = str(artifact.get("verdict") or seal.get("verdict") or "UNWRITTEN")
    overall = _number(artifact.get("overall") or seal.get("overall"), default=0)

    title_font = _font(58)
    body_font = _font(28)
    label_font = _font(20)
    small_font = _font(15)
    stamp_font = _font(27)
    score_font = _font(58)
    map_label_font = _font(18)

    _wrap_text(
        draw,
        str(artifact.get("title") or "Unwritten Page"),
        (78, 94),
        760,
        66,
        title_font,
        INK,
    )
    _wrap_text(draw, str(artifact.get("caption") or ""), (82, 235), 720, 36, body_font, INK_SOFT)
    echoes = seal.get("echoes") if isinstance(seal.get("echoes"), list) else []
    _draw_citations(draw, echoes, (742, 335), 330, small_font)

    _draw_stamp(draw, (930, 205), verdict, overall, stamp_font, score_font)
    _draw_score_bars(draw, seal, verdict, (82, 418), label_font)
    wood_map = artifact.get("wood_map") if isinstance(artifact.get("wood_map"), dict) else {}
    _draw_wood_map(draw, wood_map, (742, 465), (330, 105), verdict, map_label_font, small_font)

    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _draw_parchment(draw: ImageDraw.ImageDraw) -> None:
    for y in range(CANVAS_HEIGHT):
        t = y / max(1, CANVAS_HEIGHT - 1)
        left = _mix(PARCHMENT_LIGHT, PARCHMENT_MID, min(1, t * 1.6))
        right = _mix(PARCHMENT_MID, PARCHMENT_DARK, t)
        for x in range(CANVAS_WIDTH):
            u = x / max(1, CANVAS_WIDTH - 1)
            draw.point((x, y), fill=_mix(left, right, u))
    for i in range(360):
        x = (i * 73) % CANVAS_WIDTH
        y = (i * 37) % CANVAS_HEIGHT
        color = (59, 33, 15)
        draw.rectangle((x, y, x + 2 + (i % 7), y), fill=_blend(color, 0.16, PARCHMENT_MID))
    draw.rectangle((28, 28, CANVAS_WIDTH - 28, CANVAS_HEIGHT - 28), outline=(72, 39, 18), width=16)


def _draw_stamp(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    verdict: str,
    overall: float,
    stamp_font: ImageFont.ImageFont,
    score_font: ImageFont.ImageFont,
) -> None:
    color = GOLD if verdict.startswith("UNWRITTEN") else OXBLOOD
    x, y = center
    radius = 104
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    _wrap_text(draw, verdict, (x - 86, y - 30), 172, 32, stamp_font, CREAM, align="center")
    text = f"{overall:.1f}"
    box = draw.textbbox((0, 0), text, font=score_font)
    draw.text((x - (box[2] - box[0]) / 2, y + 4), text, font=score_font, fill=CREAM)


def _draw_score_bars(
    draw: ImageDraw.ImageDraw,
    seal: dict[str, Any],
    verdict: str,
    origin: tuple[int, int],
    font: ImageFont.ImageFont,
) -> None:
    rows = [
        ("Originality", seal.get("originality")),
        ("Delight", seal.get("delight")),
        ("AI Need", seal.get("ai_necessity")),
        ("Feasible", seal.get("feasibility")),
        ("Goal Fit", seal.get("goal_fit")),
    ]
    fill = LEAF if verdict.startswith("UNWRITTEN") else OXBLOOD
    x, y = origin
    for index, (label, value) in enumerate(rows):
        row_y = y + index * 34
        score = _number(value, default=0)
        draw.text((x, row_y - 18), label, font=font, fill=INK_SOFT)
        draw.rectangle((240, row_y - 17, 560, row_y - 1), fill=(177, 148, 111))
        draw.rectangle((240, row_y - 17, 240 + int(32 * score), row_y - 1), fill=fill)
        draw.text((582, row_y - 20), str(int(score)), font=font, fill=INK)


def _draw_wood_map(
    draw: ImageDraw.ImageDraw,
    map_data: dict[str, Any],
    origin: tuple[int, int],
    size: tuple[int, int],
    verdict: str,
    label_font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    dots = map_data.get("dots") if isinstance(map_data.get("dots"), list) else []
    if not dots:
        return
    x, y = origin
    width, height = size
    draw.rounded_rectangle(
        (x, y, x + width, y + height),
        radius=8,
        fill=(231, 205, 149),
        outline=(80, 47, 22),
        width=2,
    )
    draw.text((x, y - 26), "YOU VS THE WOOD", font=label_font, fill=INK_SOFT)

    for dot in dots:
        if not isinstance(dot, dict):
            continue
        px = x + int(width * _bounded_percent(dot.get("x")) / 100)
        py = y + int(height * _bounded_percent(dot.get("y")) / 100)
        radius = max(3, min(10, int(_number(dot.get("radius"), default=4))))
        kind = str(dot.get("kind") or "inked")
        if kind == "idea":
            color = LEAF if verdict.startswith("UNWRITTEN") else OXBLOOD
            outline = CREAM
        elif kind == "echo":
            color = OXBLOOD
            outline = CREAM
        else:
            color = (120, 88, 58)
            outline = color
        draw.ellipse(
            (px - radius, py - radius, px + radius, py + radius),
            fill=color,
            outline=outline,
            width=2,
        )

    caption = _ellipsize(str(map_data.get("caption") or ""), 78)
    _wrap_text(draw, caption, (x, y + height + 12), width, 18, small_font, INK_SOFT)


def _draw_citations(
    draw: ImageDraw.ImageDraw,
    echoes: list[Any],
    origin: tuple[int, int],
    max_width: int,
    font: ImageFont.ImageFont,
) -> None:
    if not echoes:
        return
    x, y = origin
    draw.text((x, y - 20), "CLOSEST PAGES", font=_font(18), fill=INK_SOFT)
    for index, echo in enumerate(echoes[:3]):
        if not isinstance(echo, dict):
            continue
        project = echo.get("project") if isinstance(echo.get("project"), dict) else {}
        page = echo.get("page_number") or "?"
        title = project.get("title") or project.get("id") or "Untitled"
        label = f"Page {page}: {title}"
        _wrap_text(draw, label, (x, y + 8 + index * 30), max_width, 18, font, INK_SOFT)


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    origin: tuple[int, int],
    max_width: int,
    line_height: int,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    align: str = "left",
) -> int:
    words = str(text).split()
    if not words:
        return origin[1]
    x, y = origin
    line = ""
    for word in words:
        next_line = f"{line} {word}".strip()
        if _text_width(draw, next_line, font) > max_width and line:
            _draw_line(draw, line, x, y, max_width, font, fill, align)
            line = word
            y += line_height
        else:
            line = next_line
    if line:
        _draw_line(draw, line, x, y, max_width, font, fill, align)
        y += line_height
    return y


def _draw_line(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    max_width: int,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    align: str,
) -> None:
    if align == "center":
        x = x + (max_width - _text_width(draw, text, font)) // 2
    draw.text((x, y), text, font=font, fill=fill)


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def _font(size: int) -> ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def _ellipsize(value: str, limit: int) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def _number(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _bounded_percent(value: Any) -> float:
    return max(4, min(96, _number(value, default=50)))


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = max(0, min(1, t))
    return tuple(round(a[index] * (1 - t) + b[index] * t) for index in range(3))


def _blend(
    foreground: tuple[int, int, int],
    alpha: float,
    background: tuple[int, int, int],
) -> tuple[int, int, int]:
    return _mix(background, foreground, alpha)
