import argparse
import atexit
import logging
import os
import zipfile
import io
import subprocess
import sys
import time
from urllib.parse import urljoin, urlparse
from typing import Tuple, List, Optional
import requests


logger = logging.getLogger(__name__)

DEFAULT_INPUT_VIDEO = os.path.join("data", "video", "input.mp4")
DEFAULT_OUTPUT_VIDEO = os.path.join("data", "video", "output.mp4")
DEFAULT_INPUT_IMAGE = os.path.join("data", "images", "input.png")
DEFAULT_OUTPUT_IMAGE = os.path.join("data", "images", "output.png")
DEFAULT_API_BASE_URL = os.environ.get("POSTURE_API_BASE_URL", "http://host.docker.internal:8000")
API_STARTUP_WAIT_SECONDS = 20


def _build_api_url(base_url: str, path: str) -> str:
    normalized_base = base_url.rstrip("/") + "/"
    normalized_path = path.lstrip("/")
    return urljoin(normalized_base, normalized_path)


def _normalize_base_url(base_url: str) -> str:
    """Normalizing API base URL and adding https scheme when omitted."""
    cleaned = (base_url or "").strip()
    if not cleaned:
        return ""
    if cleaned.startswith("http://") or cleaned.startswith("https://"):
        return cleaned
    return f"https://{cleaned}"


def _ensure_parent_dir(file_path: str):
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def check_posture_api_server(api_base_url: str, timeout_seconds: int = 5) -> Tuple[bool, str]:
    """Checking whether posture FastAPI service is reachable and healthy."""
    try:
        response = requests.get(_build_api_url(api_base_url, "/health"), timeout=timeout_seconds)
        if response.status_code == 200:
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                try:
                    payload = response.json()
                    status_value = str(payload.get("status", "")).lower()
                    if status_value in {"healthy", "ok"}:
                        return True, ""
                    return False, f"health JSON response is missing expected status field: {payload}"
                except Exception as exc:
                    return False, f"health endpoint returned invalid JSON: {exc}"
            return False, f"health endpoint returned unexpected content type: {content_type}"
        return False, f"health endpoint returned HTTP {response.status_code}"
    except Exception as exc:
        return False, str(exc)


def _expand_host_port_candidates(base_url: str) -> List[str]:
    """Expanding host-only base URLs to include common explicit API port candidates."""
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.hostname:
        return []

    has_explicit_port = parsed.port is not None
    has_non_root_path = bool(parsed.path and parsed.path not in {"", "/"})
    if has_explicit_port or has_non_root_path:
        return []

    host = parsed.hostname
    return [
        f"http://{host}:8000",
        f"https://{host}:8000",
    ]


def resolve_posture_api_base_url(configured_base_url: str) -> Tuple[str, bool, str]:
    """Resolving a reachable posture API base URL with local/container fallbacks."""
    normalized_configured_base_url = _normalize_base_url(configured_base_url)
    candidate_urls: List[str] = [normalized_configured_base_url]
    candidate_urls.extend(_expand_host_port_candidates(normalized_configured_base_url))

    fallback_env = os.environ.get("POSTURE_API_FALLBACK_URLS", "").strip()
    if fallback_env:
        for url in fallback_env.split(","):
            cleaned = url.strip()
            if cleaned:
                candidate_urls.append(_normalize_base_url(cleaned))
    else:
        candidate_urls.extend([
            "http://seriousbenentertainment.org:8000",
            "https://seriousbenentertainment.org:8000",
            "https://seriousbenentertainment.org",
            "http://seriousbenentertainment.org",
            "http://127.0.0.1:8000",
            "http://host.docker.internal:8000",
            "http://backend:8000",
        ])

    seen = set()
    unique_candidates: List[str] = []
    for url in candidate_urls:
        if url not in seen:
            unique_candidates.append(url)
            seen.add(url)

    errors = []
    for candidate in unique_candidates:
        ok, error = check_posture_api_server(candidate)
        if ok:
            if candidate != normalized_configured_base_url:
                return candidate, True, (
                    f"Configured API '{normalized_configured_base_url}' was unreachable. "
                    f"Using fallback '{candidate}'."
                )
            return candidate, True, ""
        errors.append(f"{candidate} -> {error}")

    return normalized_configured_base_url, False, " | ".join(errors)


def start_local_posture_api(wait_seconds: int = API_STARTUP_WAIT_SECONDS) -> Tuple[Optional[subprocess.Popen], str]:
    """Starting local FastAPI server for posture processing when not already running."""
    api_script_path = os.path.join("code", "scripts", "mediapipe_api.py")
    if not os.path.exists(api_script_path):
        return None, f"API script not found: {api_script_path}"

    try:
        process = subprocess.Popen(
            [sys.executable, api_script_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        return None, str(exc)

    started = False
    errors = []
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if process.poll() is not None:
            return None, f"API process exited early with code {process.returncode}"

        ok, error = check_posture_api_server("http://127.0.0.1:8000", timeout_seconds=2)
        if ok:
            started = True
            break

        if error:
            errors.append(error)
        time.sleep(1)

    if not started:
        process.terminate()
        return None, "Timed out waiting for local API startup. " + " | ".join(errors[-3:])

    def _stop_process_on_exit():
        try:
            if process.poll() is None:
                process.terminate()
        except Exception:
            pass

    atexit.register(_stop_process_on_exit)
    return process, ""


def process_image(input_path, output_path, api_base_url=DEFAULT_API_BASE_URL):
    """Process an image via FastAPI and write annotated output image to output_path."""
    if not output_path:
        raise ValueError("Output path must be provided to write annotated image.")

    _ensure_parent_dir(output_path)

    endpoint = _build_api_url(api_base_url, "/posture/image")
    with open(input_path, "rb") as image_file:
        files = {
            "file": (os.path.basename(input_path) or "input.png", image_file, "application/octet-stream"),
        }
        response = requests.post(endpoint, files=files, timeout=180)

    if response.status_code >= 400:
        raise RuntimeError(f"Image processing failed ({response.status_code}): {response.text}")

    with open(output_path, "wb") as output_file:
        output_file.write(response.content)

    logger.info("Annotated image saved to: %s", output_path)


def process_video(input_source, output_path, worst_image_output_path=None, api_base_url=DEFAULT_API_BASE_URL):
    """Process a video via FastAPI and write annotated video (and optional worst frame)."""
    if not isinstance(input_source, str):
        raise ValueError("Only file-based input sources are supported when using API processing.")

    if input_source.isdigit():
        raise ValueError("Webcam sources are not supported via API mode. Provide a video file path.")

    if not output_path:
        raise ValueError("Output path must be provided to write annotated video.")

    _ensure_parent_dir(output_path)
    if worst_image_output_path:
        _ensure_parent_dir(worst_image_output_path)

    endpoint = _build_api_url(api_base_url, "/posture/video")
    include_worst = bool(worst_image_output_path)

    worst_image_written = False

    with open(input_source, "rb") as video_file:
        files = {
            "file": (os.path.basename(input_source) or "input.mp4", video_file, "video/mp4"),
        }
        data = {
            "include_worst_frame": "true" if include_worst else "false",
        }
        response = requests.post(endpoint, files=files, data=data, timeout=600)

    if response.status_code >= 400:
        raise RuntimeError(f"Video processing failed ({response.status_code}): {response.text}")

    content_type = response.headers.get("content-type", "")

    if include_worst and "application/zip" in content_type:
        archive = zipfile.ZipFile(io.BytesIO(response.content))
        video_member = None
        worst_member = None

        for member in archive.namelist():
            lower_name = member.lower()
            if lower_name.endswith(".mp4") and video_member is None:
                video_member = member
            elif lower_name.endswith(".png") and worst_member is None:
                worst_member = member

        if video_member is None:
            raise RuntimeError("API zip response did not include a processed video file.")

        with archive.open(video_member) as video_data, open(output_path, "wb") as output_video:
            output_video.write(video_data.read())

        if worst_image_output_path and worst_member is not None:
            with archive.open(worst_member) as image_data, open(worst_image_output_path, "wb") as output_image:
                output_image.write(image_data.read())
            worst_image_written = True
            logger.info("Worst-posture frame saved to: %s", worst_image_output_path)
        elif worst_image_output_path:
            logger.warning("Worst-frame image requested, but API response did not include one.")
    else:
        with open(output_path, "wb") as output_video:
            output_video.write(response.content)

    if include_worst and worst_image_output_path and not worst_image_written:
        logger.info("Requesting worst-frame image via fallback API call.")
        worst_endpoint = _build_api_url(api_base_url, "/posture/video?return_worst_frame_only=true")
        with open(input_source, "rb") as video_file:
            files = {
                "file": (os.path.basename(input_source) or "input.mp4", video_file, "video/mp4"),
            }
            worst_response = requests.post(worst_endpoint, files=files, timeout=600)

        if worst_response.status_code >= 400:
            raise RuntimeError(f"Worst-frame extraction failed ({worst_response.status_code}): {worst_response.text}")

        worst_content_type = worst_response.headers.get("content-type", "")
        if "image/png" not in worst_content_type:
            raise RuntimeError(
                "Worst-frame extraction response was not a PNG image. "
                f"Received content-type: {worst_content_type}"
            )

        with open(worst_image_output_path, "wb") as output_image:
            output_image.write(worst_response.content)

        worst_image_written = True
        logger.info("Worst-posture frame saved to: %s", worst_image_output_path)

    if include_worst and worst_image_output_path and not worst_image_written:
        raise RuntimeError("Worst-frame image was requested but could not be written.")

    logger.info("Annotated video saved to: %s", output_path)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Annotate posture metrics on a video or image via FastAPI.")
    parser.add_argument(
        "--mode",
        choices=["video", "image"],
        default="video",
        help="Processing mode: 'video' for video files, 'image' for single images.",
    )
    parser.add_argument(
        "--api-base-url",
        default=DEFAULT_API_BASE_URL,
        help="Base URL of the posture FastAPI service.",
    )
    parser.add_argument(
        "--input-video",
        "-i",
        default=DEFAULT_INPUT_VIDEO,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--output-video",
        "-o",
        default=DEFAULT_OUTPUT_VIDEO,
        help="Path to the output annotated video.",
    )
    parser.add_argument(
        "--input-image",
        default=DEFAULT_INPUT_IMAGE,
        help="Path to the input image when --mode image is used.",
    )
    parser.add_argument(
        "--output-image",
        default=DEFAULT_OUTPUT_IMAGE,
        help=(
            "Path to the output image. In --mode image this is the annotated image; "
            "in --mode video this stores the worst-posture frame from the video."
        ),
    )
    parser.add_argument(
        "--no-auto-start-api",
        action="store_true",
        help="Disable automatic local API startup when the API is unreachable.",
    )
    return parser.parse_args()


def main():
    """Entry point for command line execution."""
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()

    resolved_api_base_url, api_ok, api_resolution_message = resolve_posture_api_base_url(args.api_base_url)
    local_api_process = None

    if not api_ok and not args.no_auto_start_api:
        logger.info("Posture API is unreachable. Attempting local API auto-start...")
        local_api_process, startup_error = start_local_posture_api()
        if local_api_process is not None:
            logger.info("Local API started successfully.")
            resolved_api_base_url, api_ok, api_resolution_message = resolve_posture_api_base_url(args.api_base_url)
        else:
            logger.warning("Local API auto-start failed: %s", startup_error)

    if not api_ok:
        raise SystemExit(
            "Posture API is not reachable. "
            "Set --api-base-url (or POSTURE_API_BASE_URL) to a reachable endpoint. "
            "To start local API: python code/scripts/mediapipe_api.py. "
            f"Connection attempts: {api_resolution_message}"
        )

    if api_resolution_message:
        logger.warning(api_resolution_message)

    logger.info("Using posture API: %s", resolved_api_base_url)

    try:
        if args.mode == "image":
            process_image(
                args.input_image,
                args.output_image,
                api_base_url=resolved_api_base_url,
            )
        elif args.mode == "video":
            process_video(
                args.input_video,
                args.output_video,
                worst_image_output_path=args.output_image,
                api_base_url=resolved_api_base_url,
            )
        else:
            raise ValueError(f"Unknown mode: {args.mode}")
    except Exception as exc:
        logger.exception("Processing failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
