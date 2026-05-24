"""
Higgsfield AI video generation utilities for Ralliq.

Provides helpers to generate cinematic AI videos from padel match frames
using the Higgsfield API (https://higgsfield.ai).

Authentication:
    Set ONE of the following environment variables before using this module:
      - HF_KEY="<api_key>:<api_secret>"
    OR separately:
      - HF_API_KEY="<api_key>"
      - HF_API_SECRET="<api_secret>"
"""

from __future__ import annotations

import os
from typing import Literal

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Available models (subset of Higgsfield catalogue that support image input)
# ---------------------------------------------------------------------------

SUPPORTED_MODELS: dict[str, str] = {
    "kling3_0": "Kling v3.0",
    "cinematic_studio_video": "Cinematic Studio Video",
    "cinematic_studio_video_v2": "Cinematic Studio Video V2",
    "seedance_2_0": "Seedance 2.0",
    "wan2_7": "Wan 2.7",
}

DEFAULT_MODEL = "kling3_0"

# Default prompt tailored for padel match footage
DEFAULT_PROMPT = (
    "Dynamic padel match, cinematic slow-motion, professional sports photography, "
    "intense rally, players in action, dramatic lighting"
)


def is_configured() -> bool:
    """Return True if Higgsfield API credentials are present in the environment."""
    return bool(
        os.environ.get("HF_KEY")
        or (os.environ.get("HF_API_KEY") and os.environ.get("HF_API_SECRET"))
    )


def _build_arguments(
    image_uuid: str,
    prompt: str,
    model: str,
    aspect_ratio: str,
    duration: int,
    mode: str,
) -> dict:
    """Build the arguments dict for the given model."""
    base: dict = {
        "prompt": prompt,
        "start_image": image_uuid,
        "aspect_ratio": aspect_ratio,
        "duration": duration,
    }

    if model == "kling3_0":
        base["mode"] = mode          # "std" or "pro"
        base["sound"] = "off"
    elif model in ("cinematic_studio_video", "cinematic_studio_video_v2"):
        base["slow_motion"] = False
        base["sound"] = False
    # seedance_2_0 / wan2_7 accept the same base args without extra flags

    return base


def generate_video_from_frame(
    frame: np.ndarray,
    prompt: str = DEFAULT_PROMPT,
    model: str = DEFAULT_MODEL,
    aspect_ratio: Literal["16:9", "9:16", "1:1", "4:3", "3:4"] = "16:9",
    duration: int = 5,
    mode: Literal["std", "pro"] = "std",
) -> dict:
    """
    Generate an AI video from a padel match frame using Higgsfield.

    Parameters
    ----------
    frame : np.ndarray
        RGB numpy array representing the starting frame (H × W × 3, uint8).
    prompt : str
        Text prompt that guides the video generation style and motion.
    model : str
        Higgsfield model ID (see ``SUPPORTED_MODELS``).
    aspect_ratio : str
        Output video aspect ratio.
    duration : int
        Video length in seconds (typically 5 or 10).
    mode : str
        Quality mode — ``"std"`` (faster, cheaper) or ``"pro"`` (higher quality).

    Returns
    -------
    dict
        Raw Higgsfield response.  Video URL is typically at
        ``result["videos"][0]["url"]``.

    Raises
    ------
    EnvironmentError
        If API credentials are not configured.
    ImportError
        If the ``higgsfield-client`` package is not installed.
    RuntimeError
        If the Higgsfield API returns an error.
    """
    if not is_configured():
        raise EnvironmentError(
            "Higgsfield API credentials not found. "
            "Please set HF_KEY or HF_API_KEY + HF_API_SECRET environment variables."
        )

    try:
        import higgsfield_client  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "higgsfield-client is not installed. Run: pip install higgsfield-client"
        ) from exc

    if model not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model '{model}'. "
            f"Choose one of: {list(SUPPORTED_MODELS.keys())}"
        )

    # 1. Convert numpy frame → PIL Image and upload
    pil_image = Image.fromarray(frame.astype("uint8"), mode="RGB")
    image_uuid = higgsfield_client.upload_image(pil_image)

    # 2. Build model-specific arguments
    arguments = _build_arguments(image_uuid, prompt, model, aspect_ratio, duration, mode)

    # 3. Submit and wait for the result (blocking)
    result = higgsfield_client.subscribe(model, arguments=arguments)

    return result


def extract_video_url(result: dict) -> str | None:
    """
    Extract the video URL from a Higgsfield API response.

    Tries common key shapes returned by different models.
    Returns None if no URL is found.
    """
    # Most video models return {"videos": [{"url": "..."}]}
    videos = result.get("videos")
    if videos and isinstance(videos, list) and videos[0].get("url"):
        return videos[0]["url"]

    # Fallback: some models return {"url": "..."}
    if result.get("url"):
        return result["url"]

    return None
