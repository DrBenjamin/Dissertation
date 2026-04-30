# syntax=docker/dockerfile:1.6
FROM --platform=linux/amd64 python:3.13-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Installing system dependencies
USER root
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        git \
        curl \
        wget \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libegl1 \
        libgles2 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Setting the working directory inside the container
WORKDIR /workspace/project

# Adding requirements file for offline install
ADD requirements.txt /workspace/project/requirements.txt 

# Installing Python dependencies
RUN python -m pip install --upgrade pip setuptools \
    && python -m pip install --no-cache-dir -r /workspace/project/requirements.txt \
    && python -m pip install --no-cache-dir ipykernel \
    && python -m ipykernel install --user --name=python3

# Attempt to auto-patch a known MediaPipe packaging bug where
# mediapipe/tasks/python/audio/__init__.py references a module
# name without importing it. This runs safely during image build
# if MediaPipe was installed by requirements.txt.
RUN python - <<'PY'
import importlib.util, sys, re
from pathlib import Path
spec = importlib.util.find_spec('mediapipe')
if spec is None:
    print('mediapipe not installed; skipping patch')
    sys.exit(0)
import mediapipe
mp_path = Path(mediapipe.__file__)
# Expected target: <site-packages>/mediapipe/tasks/python/audio/__init__.py
target = mp_path.parent / 'tasks' / 'python' / 'audio' / '__init__.py'
if not target.exists():
    # fallback: try parent levels in case of different layout
    for p in mp_path.parents:
        cand = p / 'tasks' / 'python' / 'audio' / '__init__.py'
        if cand.exists():
            target = cand
            break

if not target.exists():
    print('mediapipe audio __init__.py not found; skipping')
    sys.exit(0)

text = target.read_text()
# Replace absolute mediapipe.audio imports (which don't bind short names)
# with relative imports so `core`, `audio_classifier`, etc. are defined.
text_new = re.sub(r"import\s+mediapipe\.tasks\.python\.audio\.(\w+)", r"from . import \1", text)
if text_new != text:
    target.write_text(text_new)
    print('patched', target)
else:
    print('pattern not found; no change')
PY

# Copying the build context (contents of the local code/ directory)
COPY . /workspace/project/

# Starting Streamlit app on default port
CMD ["python", "-m", "streamlit", "run", "code/mediapipe_pose.py", "--server.port=8502", "--server.address=0.0.0.0"]
