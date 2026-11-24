"""
Presentation helpers for the Streamlit app: theme injection, hero section and photo page banners.
Photos live in assets/images (free Unsplash license) and are inlined as base64 so they work offline.
"""

from __future__ import annotations
import base64
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import streamlit as st

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"
IMAGES_DIR = ASSETS_DIR / "images"


@lru_cache(maxsize=None)
def image_data_uri(name: str) -> str:
    """Return a base64 data URI for assets/images/<name>.jpg, or '' if missing."""
    path = IMAGES_DIR / f"{name}.jpg"
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _load_css() -> str:
    css_path = ASSETS_DIR / "style.css"
    return css_path.read_text(encoding="utf-8") if css_path.exists() else ""


def inject_theme() -> None:
    st.markdown(f"<style>{_load_css()}</style>", unsafe_allow_html=True)


def _bg_style(image: str) -> str:
    uri = image_data_uri(image)
    return f"background-image: url('{uri}');" if uri else "background: linear-gradient(135deg, #0C1018, #151A2B);"


def render_hero(eyebrow: str, title_html: str, lede: str, chips: List[Tuple[str, str]], image: str = "hero") -> None:
    """Large landing hero. title_html may contain <em> for the gradient serif accent."""
    chip_html = "".join(f"<span class='chip'>{label} <b>{value}</b></span>" for label, value in chips)
    st.markdown(
        f"""<section class='hero' style="{_bg_style(image)}">
            <div class='eyebrow'><span class='dot'></span>{eyebrow}</div>
            <h1>{title_html}</h1>
            <p class='lede'>{lede}</p>
            <div class='chip-row'>{chip_html}</div>
        </section>""",
        unsafe_allow_html=True,
    )


def render_page_banner(eyebrow: str, title_html: str, subtitle: str, image: str) -> None:
    st.markdown(
        f"""<section class='page-banner' style="{_bg_style(image)}">
            <div class='eyebrow'><span class='dot'></span>{eyebrow}</div>
            <h1>{title_html}</h1>
            <p>{subtitle}</p>
        </section>""",
        unsafe_allow_html=True,
    )


def section_label(text: str) -> None:
    st.markdown(f"<div class='section-label'>{text}</div>", unsafe_allow_html=True)


def metric_card(title: str, value: str, subtitle: Optional[str] = "") -> None:
    st.markdown(
        f"""<div class='metric-card'>
            <div class='metric-title'>{title}</div>
            <div class='metric-value'>{value}</div>
            <div class='metric-subtitle'>{subtitle}</div>
        </div>""",
        unsafe_allow_html=True,
    )
