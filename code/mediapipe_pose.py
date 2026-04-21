import streamlit as st
import io
import os
import zipfile
import logging
import tempfile
import csv
import importlib
import requests
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
import ssl
import cv2
import numpy as np
import urllib.request
try:
    _minio_module = importlib.import_module("minio")
    Minio = getattr(_minio_module, "Minio")
except Exception:
    Minio = None
from human_posture_analysis import process_video


# Setting logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class PoseConfig:
    static_image_mode: bool = True
    model_complexity: int = 2
    enable_segmentation: bool = False
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    use_custom_model: bool = False
    custom_model_path: str | None = None
    custom_model_bytes: bytes | None = None
    custom_model_filename: str | None = None
    fallback_model_path: str | None = None
    model_type: str = 'pose'
    max_classification_results: int = 4


def get_local_fallback_model_path() -> Optional[str]:
    """Return local default classification model path if present."""
    candidate = os.path.join(os.path.dirname(__file__), "model_int8.tflite")
    if os.path.isfile(candidate):
        return candidate
    return None


# Function to read uploaded image file into OpenCV BGR format
def read_image_file(file) -> np.ndarray:
    """Reading an uploaded file-like object into a BGR OpenCV image."""
    bytes_data = file.read()
    image_array = np.frombuffer(bytes_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image file uploaded.")
    return image


# Function to build in-memory zip of annotated images
def build_zip(annotated: Dict[str, np.ndarray]) -> bytes:
    """Building an in-memory zip of annotated images for download."""
    memfile = io.BytesIO()
    with zipfile.ZipFile(memfile, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, img in annotated.items():
            ext = os.path.splitext(name)[1].lower() or ".png"
            success, buffer = cv2.imencode(
                ext if ext in [".png", ".jpg", ".jpeg"] else ".png", img)
            if not success:
                continue
            zf.writestr(
                f"annotated_{os.path.basename(name)}", buffer.tobytes())
    memfile.seek(0)
    return memfile.read()


def build_classification_csv(results_data: Dict[str, list], image_sources: Dict[str, str]) -> bytes:
    """Building classification outcomes as CSV bytes with image source paths/URLs."""
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=[
            "image_name",
            "image_source_path",
            "prediction_rank",
            "category",
            "score",
            "status",
        ],
    )
    writer.writeheader()

    for image_name in sorted(image_sources.keys()):
        source_path = image_sources.get(image_name, image_name)
        predictions = results_data.get(image_name, []) or []

        if not predictions:
            writer.writerow(
                {
                    "image_name": image_name,
                    "image_source_path": source_path,
                    "prediction_rank": "",
                    "category": "",
                    "score": "",
                    "status": "no_classification",
                }
            )
            continue

        for rank, classification in enumerate(predictions, start=1):
            category = (
                classification.get("category_name")
                or classification.get("class_name")
                or classification.get("display_name")
                or ""
            )
            writer.writerow(
                {
                    "image_name": image_name,
                    "image_source_path": source_path,
                    "prediction_rank": rank,
                    "category": category,
                    "score": f"{classification.get('score', 0.0):.6f}",
                    "status": "ok",
                }
            )

    return csv_buffer.getvalue().encode("utf-8")


def read_image_bytes(bytes_data: bytes) -> np.ndarray:
    """Reading raw image bytes into a BGR OpenCV image."""
    image_array = np.frombuffer(bytes_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image bytes.")
    return image


def get_posture_api_base_url() -> str:
    """Reading FastAPI base URL from Streamlit secrets or environment defaults."""
    default_base_url = os.environ.get("POSTURE_API_BASE_URL", "http://localhost:8000")

    if "posture_api" in st.secrets:
        return str(st.secrets["posture_api"].get("base_url", default_base_url)).strip()

    return str(st.secrets.get("POSTURE_API_BASE_URL", default_base_url)).strip()


def check_posture_api_server(api_base_url: str, timeout_seconds: int = 5) -> Tuple[bool, str]:
    """Checking whether posture FastAPI service is reachable and healthy."""
    try:
        response = requests.get(f"{api_base_url.rstrip('/')}/health", timeout=timeout_seconds)
        if response.status_code == 200:
            return True, ""
        return False, f"health endpoint returned HTTP {response.status_code}"
    except Exception as exc:
        return False, str(exc)


def resolve_posture_api_base_url() -> Tuple[str, bool, str]:
    """Resolving a reachable posture API base URL with container-friendly fallbacks."""
    configured_base = get_posture_api_base_url()

    candidate_urls: List[str] = [configured_base]
    parsed = urlparse(configured_base)
    configured_host = (parsed.hostname or "").lower()

    if configured_host in {"localhost", "127.0.0.1", ""}:
        candidate_urls.extend([
            "http://backend:8000",
            "http://host.docker.internal:8000",
        ])

    fallback_env = os.environ.get("POSTURE_API_FALLBACK_URLS", "").strip()
    if fallback_env:
        for url in fallback_env.split(","):
            cleaned = url.strip()
            if cleaned:
                candidate_urls.append(cleaned)

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
            if candidate != configured_base:
                return candidate, True, (
                    f"Configured API '{configured_base}' was unreachable. "
                    f"Using fallback '{candidate}'."
                )
            return candidate, True, ""
        errors.append(f"{candidate} -> {error}")

    return configured_base, False, " | ".join(errors)


def annotate_image_via_api(image: np.ndarray, file_name: str, api_base_url: str) -> np.ndarray:
    """Sending an image to FastAPI pose endpoint and returning annotated image."""
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Failed to encode image for API upload.")

    endpoint = f"{api_base_url.rstrip('/')}/posture/image"
    files = {
        "file": (file_name, encoded.tobytes(), "image/png"),
    }
    response = requests.post(endpoint, files=files, timeout=120)

    if response.status_code >= 400:
        raise RuntimeError(f"Pose API request failed ({response.status_code}): {response.text}")

    return read_image_bytes(response.content)


def classify_image_via_api(
    image: np.ndarray,
    file_name: str,
    model_path: str,
    model_file_bytes: Optional[bytes],
    model_file_name: Optional[str],
    min_confidence: float,
    max_results: int,
    api_base_url: str,
) -> List[Dict[str, object]]:
    """Sending an image to FastAPI classification endpoint and returning category list."""
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Failed to encode image for API upload.")

    endpoint = f"{api_base_url.rstrip('/')}/posture/classify"
    files = {
        "file": (file_name, encoded.tobytes(), "image/png"),
    }
    if model_file_bytes is not None:
        files["model_file"] = (
            model_file_name or "model.tflite",
            model_file_bytes,
            "application/octet-stream",
        )

    data = {
        "model_path": model_path,
        "min_confidence": str(min_confidence),
        "max_results": str(max_results),
    }
    response = requests.post(endpoint, files=files, data=data, timeout=120)

    if response.status_code >= 400:
        raise RuntimeError(f"Classification API request failed ({response.status_code}): {response.text}")

    payload = response.json()
    classifications = payload.get("classifications", [])
    if isinstance(classifications, list):
        return classifications
    return []


def _parse_bool(value, default=False) -> bool:
    """Parsing MinIO secure flag from bool-like values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    if value is None:
        return default
    return bool(value)


def _normalize_minio_endpoint(endpoint: str, secure_default: bool):
    """Normalizing MinIO endpoint and secure flag for Minio client."""
    endpoint_str = str(endpoint).strip()
    parsed = urlparse(endpoint_str)

    if parsed.scheme in {"http", "https"} and parsed.netloc:
        normalized_endpoint = parsed.netloc
        secure = parsed.scheme == "https"
        return normalized_endpoint, secure

    normalized_endpoint = endpoint_str.replace("https://", "").replace("http://", "")
    return normalized_endpoint, secure_default


def _normalize_minio_prefix(prefix: str) -> str:
    """Normalizing MinIO object prefix by trimming extra slashes."""
    if not prefix:
        return ""
    normalized = str(prefix).strip().strip("/")
    return normalized


def get_minio_config() -> Dict[str, object]:
    """Reading MinIO configuration from Streamlit secrets."""
    endpoint = None
    access_key = None
    secret_key = None
    secure = False

    if "minio" in st.secrets:
        minio_cfg = st.secrets["minio"]
        endpoint = minio_cfg.get("endpoint")
        access_key = minio_cfg.get("access_key")
        secret_key = minio_cfg.get("secret_key")
        secure = _parse_bool(minio_cfg.get("secure", False), default=False)
    else:
        endpoint = st.secrets.get("MINIO_ENDPOINT")
        access_key = st.secrets.get("MINIO_ACCESS_KEY")
        secret_key = st.secrets.get("MINIO_SECRET_KEY")
        secure = _parse_bool(st.secrets.get("MINIO_SECURE", False), default=False)

    if not endpoint or not access_key or not secret_key:
        return {}

    normalized_endpoint, normalized_secure = _normalize_minio_endpoint(
        endpoint,
        secure_default=secure,
    )

    return {
        "endpoint": normalized_endpoint,
        "access_key": access_key,
        "secret_key": secret_key,
        "secure": normalized_secure,
    }


def _build_minio_url(endpoint: str, secure: bool, path: str = "/") -> str:
    """Building MinIO URL from normalized endpoint and secure flag."""
    scheme = "https" if secure else "http"
    normalized_path = path if path.startswith("/") else f"/{path}"
    return f"{scheme}://{endpoint}{normalized_path}"


def check_minio_server(endpoint: str, secure: bool, allow_insecure_tls: bool = False):
    """Checking MinIO server reachability and TLS validity.

    Returns: (ok, error, cert_issue)
    """
    url = _build_minio_url(endpoint, secure, "/minio/health/live")
    context = None
    if secure and allow_insecure_tls:
        context = ssl._create_unverified_context()

    try:
        with urllib.request.urlopen(url, timeout=5, context=context) as response:
            status = getattr(response, "status", 200)
            if 200 <= status < 500:
                return True, "", False
            return False, f"Unexpected HTTP status: {status}", False
    except Exception as exc:
        error_text = str(exc)
        cert_issue = (
            "CERTIFICATE_VERIFY_FAILED" in error_text
            or "certificate verify failed" in error_text.lower()
            or isinstance(exc, ssl.SSLCertVerificationError)
            or isinstance(getattr(exc, "reason", None), ssl.SSLCertVerificationError)
        )
        return False, error_text, cert_issue


def create_minio_client(
    endpoint: str,
    access_key: str,
    secret_key: str,
    secure: bool,
    allow_insecure_tls: bool = False,
):
    """Creating MinIO client with optional insecure TLS mode."""
    kwargs = {
        "access_key": access_key,
        "secret_key": secret_key,
        "secure": secure,
    }

    if secure and allow_insecure_tls:
        urllib3 = importlib.import_module("urllib3")
        kwargs["http_client"] = urllib3.PoolManager(
            cert_reqs=ssl.CERT_NONE,
            assert_hostname=False,
        )

    return Minio(endpoint, **kwargs)


@st.cache_data(show_spinner=False, ttl=30)
def list_minio_buckets(
    endpoint: str,
    access_key: str,
    secret_key: str,
    secure: bool,
    allow_insecure_tls: bool = False,
):
    """Listing available MinIO buckets."""
    if Minio is None:
        return [], "MinIO client library not installed. Please install package 'minio'."

    try:
        client = create_minio_client(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
            allow_insecure_tls=allow_insecure_tls,
        )
        bucket_names = sorted([bucket.name for bucket in client.list_buckets()])
        return bucket_names, ""
    except Exception as exc:
        return [], str(exc)


@st.cache_data(show_spinner=False, ttl=30)
def list_minio_image_objects(
    endpoint: str,
    access_key: str,
    secret_key: str,
    secure: bool,
    bucket_name: str,
    prefix: str = "",
    allow_insecure_tls: bool = False,
) -> Tuple[List[str], str]:
    """Listing image object keys from a MinIO bucket/prefix."""
    if Minio is None:
        return [], "MinIO client library not installed. Please install package 'minio'."

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    try:
        client = create_minio_client(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
            allow_insecure_tls=allow_insecure_tls,
        )

        normalized_prefix = _normalize_minio_prefix(prefix)
        object_iter = client.list_objects(
            bucket_name,
            prefix=normalized_prefix if normalized_prefix else None,
            recursive=True,
        )

        image_keys = []
        for obj in object_iter:
            object_name = getattr(obj, "object_name", "")
            if not object_name:
                continue
            ext = os.path.splitext(object_name)[1].lower()
            if ext in image_extensions:
                image_keys.append(object_name)

        return sorted(image_keys), ""
    except Exception as exc:
        return [], str(exc)


def load_minio_images(
    endpoint: str,
    access_key: str,
    secret_key: str,
    secure: bool,
    bucket_name: str,
    object_keys: List[str],
    allow_insecure_tls: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, str], List[str], List[str]]:
    """Loading MinIO objects as OpenCV images.

    Returns images, image_sources, duplicate_names, failed_objects.
    """
    images: Dict[str, np.ndarray] = {}
    image_sources: Dict[str, str] = {}
    duplicate_names: List[str] = []
    failed_objects: List[str] = []

    if Minio is None:
        return images, image_sources, duplicate_names, failed_objects

    client = create_minio_client(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure,
        allow_insecure_tls=allow_insecure_tls,
    )

    for object_key in object_keys:
        display_name = os.path.basename(object_key)
        if display_name in images:
            duplicate_names.append(display_name)
            continue

        response = None
        try:
            response = client.get_object(bucket_name, object_key)
            image_bytes = response.read()
            images[display_name] = read_image_bytes(image_bytes)
            image_sources[display_name] = f"minio://{bucket_name}/{object_key}"
        except Exception:
            failed_objects.append(object_key)
        finally:
            if response is not None:
                response.close()
                response.release_conn()

    return images, image_sources, duplicate_names, failed_objects


def process_uploaded_video_file(uploaded_file) -> Dict[str, object]:
    """Processing an uploaded video and returning annotated video and worst-frame data."""
    suffix = os.path.splitext(uploaded_file.name)[1].lower() or ".mp4"

    with tempfile.TemporaryDirectory(prefix="posture_video_") as temp_dir:
        input_path = os.path.join(temp_dir, f"input{suffix}")
        output_path = os.path.join(temp_dir, "annotated_output.mp4")
        worst_frame_path = os.path.join(temp_dir, "worst_posture_frame.png")

        with open(input_path, "wb") as temp_input:
            temp_input.write(uploaded_file.read())

        process_video(
            input_path,
            output_path,
            worst_image_output_path=worst_frame_path,
        )

        if not os.path.exists(output_path):
            raise FileNotFoundError("Annotated video output was not created.")

        with open(output_path, "rb") as output_video:
            annotated_video_bytes = output_video.read()

        worst_frame_image = None
        worst_frame_bytes = None
        if os.path.exists(worst_frame_path):
            worst_frame_image = cv2.imread(worst_frame_path)
            if worst_frame_image is not None:
                ok, encoded = cv2.imencode(".png", worst_frame_image)
                if ok:
                    worst_frame_bytes = encoded.tobytes()

        return {
            "name": uploaded_file.name,
            "annotated_video_bytes": annotated_video_bytes,
            "worst_frame_image": worst_frame_image,
            "worst_frame_bytes": worst_frame_bytes,
        }


# Function to render sidebar controls and return configuration
def sidebar_configuration() -> PoseConfig:
    """Rendering sidebar controls for MediaPipe Pose parameters."""
    st.sidebar.header("Model Configuration")
    model_type = 'pose'
    
    # Model selection section
    st.sidebar.subheader("Model Selection")
    use_custom_model = st.sidebar.checkbox(
        "Use Custom TensorFlow Lite Model", 
        value=False,
        help="Enable to use a custom trained TensorFlow Lite model"
    )

    custom_model_path = None
    custom_model_bytes = None
    custom_model_filename = None
    fallback_model_path = get_local_fallback_model_path()

    if use_custom_model:
        model_type = st.sidebar.selectbox(
            "Model Type",
            options=['pose', 'classification'],
            index=0,
            help="Select whether this model is for pose detection or image classification",
        )

        uploaded_model = st.sidebar.file_uploader(
            "Upload Custom TensorFlow Lite Model",
            type=['tflite'],
            help="Upload your custom trained .tflite model file",
        )

        if uploaded_model is not None:
            model_bytes = uploaded_model.read()
            temp_model_path = f"/tmp/{uploaded_model.name}"
            with open(temp_model_path, "wb") as f:
                f.write(model_bytes)
            custom_model_path = temp_model_path
            custom_model_bytes = model_bytes
            custom_model_filename = uploaded_model.name
            st.sidebar.success("✅ Custom model file uploaded.")

        if custom_model_path is None:
            model_path_input = st.sidebar.text_input(
                "Or specify model file path:",
                placeholder="../data/models/efficientnet_lite2/20260403_162843/model_int8.tflite",
                help="Enter the full path to your custom TensorFlow Lite model file on the API host",
            )
            if model_path_input:
                custom_model_path = model_path_input

        if model_type == 'classification' and not custom_model_path:
            if fallback_model_path:
                st.sidebar.info(f"Using local fallback model: {fallback_model_path}")
            else:
                st.sidebar.warning("⚠️ Classification mode needs a model path/upload, or local files/model_int8.tflite.")

    minio_config = get_minio_config()
    selected_minio_bucket = None

    if minio_config:
        st.sidebar.subheader("MinIO Bucket")

        allow_insecure_tls = st.session_state.get("minio_allow_insecure_tls", False)
        should_list_buckets = True
        server_ok, server_error, cert_issue = check_minio_server(
            minio_config["endpoint"],
            bool(minio_config["secure"]),
            allow_insecure_tls=allow_insecure_tls,
        )

        if not server_ok and cert_issue and not allow_insecure_tls:
            insecure_probe_ok, insecure_probe_error, _ = check_minio_server(
                minio_config["endpoint"],
                bool(minio_config["secure"]),
                allow_insecure_tls=True,
            )
            if insecure_probe_ok:
                st.sidebar.warning(
                    "MinIO server is reachable, but TLS certificate verification failed. "
                    "You can enable insecure TLS below as a temporary workaround."
                )
                allow_insecure_tls = st.sidebar.checkbox(
                    "Allow insecure TLS for MinIO",
                    value=False,
                    help="Disables TLS certificate verification for MinIO requests.",
                )
                st.session_state["minio_allow_insecure_tls"] = allow_insecure_tls

                if not allow_insecure_tls:
                    should_list_buckets = False
                    st.sidebar.info(
                        "Bucket listing is paused until insecure TLS is enabled or TLS certificates are fixed on the server."
                    )
            else:
                st.sidebar.warning(
                    f"Could not verify MinIO server securely ({server_error}) and insecure probe also failed ({insecure_probe_error})."
                )
                st.session_state["minio_allow_insecure_tls"] = False
                should_list_buckets = False
        elif not server_ok:
            st.sidebar.warning(f"Could not reach MinIO server: {server_error}")
            st.session_state["minio_allow_insecure_tls"] = allow_insecure_tls
            should_list_buckets = False
        else:
            if allow_insecure_tls:
                st.sidebar.info("Insecure TLS mode is enabled for MinIO.")
            st.session_state["minio_allow_insecure_tls"] = allow_insecure_tls

        if should_list_buckets:
            bucket_options, bucket_error = list_minio_buckets(
                minio_config["endpoint"],
                minio_config["access_key"],
                minio_config["secret_key"],
                bool(minio_config["secure"]),
                allow_insecure_tls=allow_insecure_tls,
            )
        else:
            bucket_options, bucket_error = [], ""

        if bucket_error:
            st.sidebar.warning(f"Could not load MinIO buckets: {bucket_error}")
        elif bucket_options:
            selected_minio_bucket = st.sidebar.selectbox(
                "Select MinIO bucket",
                options=bucket_options,
                index=0,
                help="Choose the bucket used to build source paths for classification exports.",
            )
            st.session_state["selected_minio_bucket"] = selected_minio_bucket

            selected_minio_prefix = st.sidebar.text_input(
                "MinIO sub-folder (optional)",
                value=st.session_state.get("selected_minio_prefix", ""),
                placeholder="e.g. posture/normal",
                help="Optional object prefix/sub-folder inside the selected bucket.",
            )
            st.session_state["selected_minio_prefix"] = _normalize_minio_prefix(
                selected_minio_prefix
            )

            export_source_mode = st.sidebar.selectbox(
                "Classification export source",
                options=[
                    "Uploaded file paths",
                    "Selected MinIO bucket paths",
                ],
                index=0,
                help=(
                    "Choose whether CSV exports use the uploaded file names/paths "
                    "or synthetic minio://bucket/object paths."
                ),
            )
            st.session_state["classification_export_source_mode"] = export_source_mode
        else:
            st.sidebar.info("No MinIO buckets found for configured server.")
            st.session_state.pop("selected_minio_bucket", None)
            st.session_state.pop("selected_minio_prefix", None)
            st.session_state.pop("classification_export_source_mode", None)
    else:
        st.session_state.pop("minio_allow_insecure_tls", None)
        st.session_state.pop("selected_minio_bucket", None)
        st.session_state.pop("selected_minio_prefix", None)
        st.session_state.pop("classification_export_source_mode", None)

    # Configuration based on model type
    if model_type == 'classification':
        st.sidebar.subheader("Image Classification Configuration")
        max_results = st.sidebar.slider("Max classification results", 1, 10, 4, 1)
        min_detection_confidence = st.sidebar.slider(
            "Min classification confidence", 0.0, 1.0, 0.1, 0.05)
        
        # Set defaults for unused pose parameters
        static_image_mode = True
        model_complexity = 2
        enable_segmentation = False
        min_tracking_confidence = 0.5
        
    else:
        # Standard MediaPipe configuration for pose detection
        st.sidebar.subheader("Pose Detection Configuration")
        
        # Only show MediaPipe options if not using custom model
        if not use_custom_model or custom_model_path is None:
            static_image_mode = st.sidebar.checkbox("Static image mode", value=True,
                                                    help="If False enables tracking across frames (for video).")
            model_complexity = st.sidebar.selectbox("Model complexity", options=[0, 1, 2], index=2,
                                                    help="0=lite, 1=full, 2=heavy for higher accuracy.")
            enable_segmentation = st.sidebar.checkbox(
                "Enable segmentation", value=False)
        else:
            # Use default values for custom models
            static_image_mode = True
            model_complexity = 2
            enable_segmentation = False
            st.sidebar.info("MediaPipe settings are not applicable when using custom pose models.")
        
        # Confidence thresholds for pose detection
        min_detection_confidence = st.sidebar.slider(
            "Min detection confidence", 0.0, 1.0, 0.5, 0.05)
        min_tracking_confidence = st.sidebar.slider(
            "Min tracking confidence", 0.0, 1.0, 0.5, 0.05)
        max_results = 4  # Default for pose detection

    # Returning configuration dataclass
    return PoseConfig(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        enable_segmentation=enable_segmentation,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        use_custom_model=use_custom_model,
        custom_model_path=custom_model_path,
        custom_model_bytes=custom_model_bytes,
        custom_model_filename=custom_model_filename,
        fallback_model_path=fallback_model_path,
        model_type=model_type,
        max_classification_results=max_results,
    )


# Main application function
def main():
    """Running the Streamlit application."""
    st.set_page_config(
        page_title="MediaPipe Pose & Image Classification",
        page_icon="🧘",
        layout="wide"
    )

    # Application title and description
    st.title("🧘 MediaPipe Pose Detection & Image Classification")
    st.markdown("Upload images to analyze human pose or classify images using MediaPipe or custom TensorFlow Lite models.")

    # Configuring detection parameters
    cfg = sidebar_configuration()

    # Displaying current model information
    if cfg.use_custom_model and cfg.custom_model_path:
        if cfg.model_type == 'classification':
            st.info(f"🏷️ Using custom TensorFlow Lite image classification model: `{os.path.basename(cfg.custom_model_path)}`")
        else:
            st.info(f"🤖 Using custom TensorFlow Lite pose model: `{os.path.basename(cfg.custom_model_path)}`")
    else:
        st.info(f"🎯 Using standard MediaPipe Pose (Complexity: {cfg.model_complexity})")

    if cfg.model_type == 'classification' and not cfg.custom_model_path and cfg.fallback_model_path:
        st.info(f"🏷️ Using local fallback classification model: `{os.path.basename(cfg.fallback_model_path)}`")

    api_base_url, api_ok, api_resolution_message = resolve_posture_api_base_url()
    if api_ok:
        st.caption(f"Using posture API: {api_base_url}")
        if api_resolution_message:
            st.warning(api_resolution_message)
    else:
        st.error(
            "Posture API is not reachable. "
            "Please verify POSTURE_API_BASE_URL (or posture_api.base_url in Streamlit secrets), "
            "and ensure the API container is running and exposing port 8000. "
            f"Connection attempts: {api_resolution_message}"
        )
        return

    # Upload section
    is_video_mode = cfg.model_type == "pose" and not cfg.static_image_mode

    if cfg.model_type == "classification":
        upload_label = "Upload image files"
        upload_types = ["png", "jpg", "jpeg", "bmp"]
        upload_help = "Upload one or more images for classification"
    elif is_video_mode:
        upload_label = "Upload video files"
        upload_types = ["mp4", "mov", "avi", "mkv"]
        upload_help = "Upload one or more videos containing people for posture analysis"
    else:
        upload_label = "Upload image files"
        upload_types = ["png", "jpg", "jpeg", "bmp"]
        upload_help = "Upload one or more images containing people for pose analysis"

    uploaded_files = st.file_uploader(
        upload_label,
        type=upload_types,
        accept_multiple_files=True,
        help=upload_help
    )

    use_minio_files = False
    if not is_video_mode:
        selected_minio_bucket = st.session_state.get("selected_minio_bucket")
        if selected_minio_bucket:
            selected_minio_prefix = st.session_state.get("selected_minio_prefix", "")
            allow_insecure_tls = st.session_state.get("minio_allow_insecure_tls", False)
            minio_config = get_minio_config()

            if minio_config:
                preview_keys, preview_error = list_minio_image_objects(
                    minio_config["endpoint"],
                    minio_config["access_key"],
                    minio_config["secret_key"],
                    bool(minio_config["secure"]),
                    selected_minio_bucket,
                    prefix=selected_minio_prefix,
                    allow_insecure_tls=allow_insecure_tls,
                )

                if preview_error:
                    st.warning(f"MinIO preview error: {preview_error}")
                else:
                    if selected_minio_prefix:
                        st.caption(
                            f"MinIO preview: {len(preview_keys)} image file(s) found in '{selected_minio_bucket}/{selected_minio_prefix}'."
                        )
                    else:
                        st.caption(
                            f"MinIO preview: {len(preview_keys)} image file(s) found in '{selected_minio_bucket}'."
                        )

            use_minio_files = st.button(
                "Upload MinIO files",
                help="Load and process image files directly from the selected MinIO bucket/prefix.",
            )

    # Handling no uploads
    if not uploaded_files and not use_minio_files:
        if is_video_mode:
            st.info("Please upload one or more video files to begin.")
        else:
            st.info("Please upload one or more image files to begin.")
        
        # Adding information about custom model training
        if cfg.model_type == 'classification':
            with st.expander("📝 Train Your Own Custom Image Classification Model"):
                st.markdown("""
                **Want to create a custom image classification model?**
                
                1. Use the `custom_image_classifier_model_training.ipynb` notebook in the `code/` directory
                2. Prepare your training data with labeled images
                3. Train and export your custom TensorFlow Lite model
                4. Upload or specify the path to your `.tflite` model file using the sidebar
                
                Custom image classification models can be specialized for specific use cases, 
                objects, or domains that may not be well covered by standard models.
                """)
        else:
            with st.expander("📝 Train Your Own Custom Pose Model"):
                st.markdown("""
                **Want to create a custom pose detection model?**
                
                1. Use the `custom_pose_model_training.ipynb` notebook in the `code/` directory
                2. Prepare your training data with pose annotations
                3. Train and export your custom TensorFlow Lite model
                4. Upload or specify the path to your `.tflite` model file using the sidebar
                
                Custom models can be specialized for specific use cases, poses, or populations 
                that may not be well covered by the standard MediaPipe models.
                """)
        return

    if is_video_mode:
        # Processing videos with worst-posture-frame extraction
        video_results = []
        with st.spinner("Processing videos with pose detection..."):
            for file in uploaded_files:
                try:
                    video_results.append(process_uploaded_video_file(file))
                except Exception as e:
                    st.error(f"Failed to process {file.name}: {e}")

        if not video_results:
            st.error("No valid videos to process.")
            return

        st.subheader("Video Pose Detection Results")
        for result in video_results:
            st.markdown(f"**{result['name']}**")
            st.video(result["annotated_video_bytes"])

            if result["worst_frame_image"] is not None:
                st.markdown("**Worst posture frame**")
                st.image(cv2.cvtColor(result["worst_frame_image"], cv2.COLOR_BGR2RGB), width='stretch')

            base_name, _ = os.path.splitext(result["name"])
            st.download_button(
                label=f"Download annotated video ({result['name']})",
                data=result["annotated_video_bytes"],
                file_name=f"annotated_{base_name}.mp4",
                mime="video/mp4",
            )

            if result["worst_frame_bytes"] is not None:
                st.download_button(
                    label=f"Download worst posture frame (worst_posture_{base_name}.png)",
                    data=result["worst_frame_bytes"],
                    file_name=f"worst_posture_{base_name}.png",
                    mime="image/png",
                )
            st.markdown("---")
    else:
        # Creating image dict
        images = {}
        image_sources = {}
        skipped_duplicate_names = []

        if uploaded_files:
            for file in uploaded_files:
                display_name = os.path.basename(file.name)
                if display_name in images:
                    skipped_duplicate_names.append(display_name)
                    continue
                try:
                    images[display_name] = read_image_file(file)
                    selected_minio_bucket = st.session_state.get("selected_minio_bucket")
                    export_source_mode = st.session_state.get(
                        "classification_export_source_mode",
                        "Uploaded file paths",
                    )
                    if selected_minio_bucket and export_source_mode == "Selected MinIO bucket paths":
                        selected_minio_prefix = st.session_state.get("selected_minio_prefix", "")
                        object_key = display_name
                        if selected_minio_prefix:
                            object_key = f"{selected_minio_prefix}/{display_name}"
                        image_sources[display_name] = f"minio://{selected_minio_bucket}/{object_key}"
                    else:
                        image_sources[display_name] = file.name
                except ValueError as e:
                    st.error(f"Failed to read {display_name}: {e}")
        elif use_minio_files:
            selected_minio_bucket = st.session_state.get("selected_minio_bucket")
            selected_minio_prefix = st.session_state.get("selected_minio_prefix", "")
            allow_insecure_tls = st.session_state.get("minio_allow_insecure_tls", False)
            minio_config = get_minio_config()

            if not selected_minio_bucket or not minio_config:
                st.error("MinIO is not configured correctly or no bucket is selected.")
                return

            object_keys, object_error = list_minio_image_objects(
                minio_config["endpoint"],
                minio_config["access_key"],
                minio_config["secret_key"],
                bool(minio_config["secure"]),
                selected_minio_bucket,
                prefix=selected_minio_prefix,
                allow_insecure_tls=allow_insecure_tls,
            )

            if object_error:
                st.error(f"Failed to list MinIO objects: {object_error}")
                return

            if not object_keys:
                if selected_minio_prefix:
                    st.warning(
                        f"No image files found in bucket '{selected_minio_bucket}' with prefix '{selected_minio_prefix}'."
                    )
                else:
                    st.warning(f"No image files found in bucket '{selected_minio_bucket}'.")
                return

            (
                images,
                image_sources,
                duplicate_names,
                failed_objects,
            ) = load_minio_images(
                minio_config["endpoint"],
                minio_config["access_key"],
                minio_config["secret_key"],
                bool(minio_config["secure"]),
                selected_minio_bucket,
                object_keys,
                allow_insecure_tls=allow_insecure_tls,
            )

            skipped_duplicate_names.extend(duplicate_names)

            if failed_objects:
                st.warning(
                    "Failed to load MinIO objects: " + ", ".join(failed_objects[:10])
                    + (" ..." if len(failed_objects) > 10 else "")
                )

        if skipped_duplicate_names:
            unique_duplicates = sorted(set(skipped_duplicate_names))
            st.warning(
                "Skipped duplicate filenames (same basename): "
                + ", ".join(unique_duplicates)
            )

        if not images:
            st.error("No valid images to process.")
            return

        # Processing images via API
        processing_text = "image classification" if cfg.model_type == 'classification' else "pose detection"
        annotated = {}
        results_data = {}

        with st.spinner(f"Processing images with {processing_text} via API..."):
            for name, img in images.items():
                try:
                    if cfg.model_type == 'classification':
                        effective_model_path = cfg.custom_model_path or cfg.fallback_model_path or ""
                        if not effective_model_path and cfg.custom_model_bytes is None:
                            raise ValueError(
                                "Classification mode requires a model path/upload, or files/model_int8.tflite."
                            )

                        classifications = classify_image_via_api(
                            image=img,
                            file_name=name,
                            model_path=effective_model_path,
                            model_file_bytes=cfg.custom_model_bytes,
                            model_file_name=cfg.custom_model_filename,
                            min_confidence=cfg.min_detection_confidence,
                            max_results=cfg.max_classification_results,
                            api_base_url=api_base_url,
                        )
                        results_data[name] = classifications
                        annotated[name] = img.copy()
                    else:
                        annotated[name] = annotate_image_via_api(
                            image=img,
                            file_name=name,
                            api_base_url=api_base_url,
                        )
                        results_data[name] = {}
                except Exception as exc:
                    st.error(f"Failed to process {name}: {exc}")
                    annotated[name] = img.copy()
                    results_data[name] = [] if cfg.model_type == 'classification' else {}

        if not annotated:
            st.error("No images were successfully processed by the API.")
            return

        # Displaying results based on model type
        if cfg.model_type == 'classification':
            st.subheader("Classification Results")
            cols = st.columns(min(3, len(annotated)))
            for idx, (name, img) in enumerate(annotated.items()):
                col = cols[idx % len(cols)]
                with col:
                    st.markdown(f"**{name}**")
                    # Converting BGR to RGB for display
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width='stretch')

                    # Showing classification results
                    classifications = results_data.get(name, [])
                    if isinstance(classifications, list) and classifications:
                        st.markdown("**Classifications:**")
                        for i, classification in enumerate(classifications[:3]):  # Show top 3
                            category = (
                                classification.get('category_name')
                                or classification.get('class_name')
                                or classification.get('display_name')
                                or f'Class_{classification.get("index", i)}'
                            )
                            score = classification.get('score', 0.0)
                            st.write(f"{i+1}. {category}: {score:.3f}")
                    else:
                        st.write("No classifications found")

            csv_bytes = build_classification_csv(results_data, image_sources)
            st.download_button(
                label="Download classification outcomes (CSV)",
                data=csv_bytes,
                file_name="classification_outcomes.csv",
                mime="text/csv",
            )
        else:
            st.subheader("Pose Detection Results")
            cols = st.columns(min(3, len(annotated)))
            for idx, (name, img) in enumerate(annotated.items()):
                col = cols[idx % len(cols)]
                with col:
                    st.markdown(f"**{name}**")
                    # Converting BGR to RGB for display
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width='stretch')
                    if results_data.get(name, {}).get("nose"):
                        nx, ny = results_data[name]["nose"]
                        st.caption(f"Nose (px): ({nx:.1f}, {ny:.1f})")

            st.caption("Pose overlays rendered by FastAPI service.")

        # Downloading zip of annotated images
        zip_bytes = build_zip(annotated)
        st.download_button(
            label="Download annotated images (zip)",
            data=zip_bytes,
            file_name="annotated_images.zip",
            mime="application/zip",
        )
    st.markdown("---")

    # Model information and future work
    with st.expander("ℹ️ About this Application"):
        if cfg.use_custom_model and cfg.custom_model_path:
            if cfg.model_type == 'classification':
                st.markdown(f"""
                **Custom TensorFlow Lite Image Classification Mode**

                You are using a custom trained image classification model. This allows for:
                - Specialized image classification for specific use cases
                - Domain-specific object or scene recognition
                - Optimized performance for your target classification scenarios

                Model file: `{cfg.custom_model_path}`
                Max results: {cfg.max_classification_results}
                """)
            else:
                st.markdown(f"""
                **Custom TensorFlow Lite Pose Detection Mode**

                You are using a custom trained pose detection model. This allows for:
                - Specialized pose detection for specific use cases
                - Optimized performance for your target scenarios
                - Domain-specific pose analysis

                Model file: `{cfg.custom_model_path}`
                """)
        else:
            st.markdown(f"""
            **Standard MediaPipe Pose Mode**

            Using Google's pre-trained MediaPipe Pose model with:
            - 33 body landmark detection
            - Real-time performance
            - Robust pose estimation across various conditions
            
            Model complexity: {cfg.model_complexity} (0=lite, 1=full, 2=heavy)
            """)
    st.markdown(
        "Future work: add video processing, CSV export of all landmarks, and comparative analytics."
    )


# Running the application
if __name__ == "__main__":
    main()
