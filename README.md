# Dissertation

This is the Dissertation project at the **University of Edinburgh** in the studies **Data Science for Health and Social Care**. Please use [s2616861@ed.ac.uk](mailto:s2616861@ed.ac.uk) for any questions or inquiries about the project.

It uses a [Kaggle image dataset](https://www.kaggle.com/datasets/melsmm/posture-keypoints-detection/code) with 300 high quality images which was collected and used by a student in his final project of the *Deep Learning* School course at **MIPTa** - [Github repo](https://github.com/MakhmudovMels/posture-keypoints-detection).

**Project structure:**

```text
.
├── .github/
│   └── copilot-instructions.md
├── .gitignore
├── Dissertation.qmd
├── LICENSE
├── Proposal.md
├── README.md
├── README_Kaggle.md
├── code/
│   ├── Jenkinsfile
│   ├── Dockerfile
│   ├── Dockerfile_mediapipe
│   ├── custom_image_classifier_model_training.ipynb
│   ├── custom_pose_model_training.ipynb
│   ├── docker-compose.yml
│   ├── docker-compose_frontend_backend.yml
│   ├── environment.yml
│   ├── human_posture_analysis.ipynb
│   ├── human_posture_analysis.py
│   ├── image_classification.ipynb
│   ├── mediapipe_pose.py
│   ├── requirements.txt
│   ├── scripts/
│       ├── custom_tflite_image_classifier.py
│       ├── custom_tflite_pose.py
│       ├── mediapipe_api.py
│       └── ...
│   └── posture-keypoints-detection/
|       ├── inference.ipynb
|       ├── train.ipynb
|       ├── docker-compose.yml
│       ├── README_en.md    
|       ├── README.md
│       ├── frontend/
│           ├── Dockerfile
│           ├── main_app.py
│           └── requirements.txt
│       ├── backend/
│           ├── best.pt
│           ├── Dockerfile
│           ├── main_api.py
│           └── requirements.txt
|       ├── images/
|           └── ...
|       └── models/
|           ├── best.pt
|           └── yolo11s-pose.pt
├── docs/
│   ├── Bachelor Thesis/
│   │   └── Bachelor.Thesis.pdf
│   ├── Ethics/
│   │   ├── Data Management Flow Chart.docx
│   │   ├── Gross_UMREG_Ethical_Considerations_Form_Export.pdf
│   │   ├── Gross_UMREG_Ethical_Considerations_Form.docx
│   │   ├── Introduction to Research Integrity Resources.pdf
│   │   ├── Local Ethics approval.pdf
│   │   └── MSc Data Science for Health and Social Care - UMREG Application Form - 2025:26 AY.pdf
│   ├── Proposal/
│       ├── Dissertation_proposal_B248593_converted.qmd
│       ├── Dissertation_proposal_B248593.docx
│       ├── Dissertation_proposal_B248593.pdf
│       ├── MSc_DSHSC_Supervisor_Appraisal_of_Project_Risk.docx
│       ├── MSc_DSHSC_Supervisor_Appraisal_of_Project_Risk.pdf
│       └── receipt_Dissertation_proposal_B248593.pdf.pdf
│   ├── Reflective Blog/
│       ├── docs/Reflective Blog/B248593_Blog_1_instructions.qmd
│       ├── B248593_Blog_1.qmd
│       ├── Reflective_Blog.qmd
│       └── Reflective_Writing.pdf
│   ├── MSc_DSHSC_Supervision_research_meeting_diary_template.docx
│   └── ...
└── literature/
    ├── apa.csl
    ├── bibliography.bib
    └── ...
```

<!-- markdownlint-disable MD051 -->
- Research Proposal for the dissertation (Word document): [Dissertation_proposal_B248593.docx](docs/Proposal/Dissertation_proposal_B248593.docx)
- Dissertation (Quarto file): [Dissertation.qmd](Dissertation.qmd)
- Reflective Blog 1: [B248593_Blog_1.qmd](docs/Reflective Blog/B248593_Blog_1.qmd)
- README (Markdown): [README.md](README.md)
- LICENSE.txt (Creative Commons Attribution 4.0 International Public License))
- Code (folder): [code](code/)
  - Jenkins pipeline configuration file: [Jenkinsfile](code/Jenkinsfile)
  - Docker Compose file for setting up the environment: [docker-compose.yml](code/docker-compose.yml)
  - Docker Compose file for setting up the frontend and backend services: [docker-compose_frontend_backend.yml](code/docker-compose_frontend_backend.yml)
  - Dockerfile for the Streamlit app environment: [Dockerfile](code/Dockerfile)
  - Dockerfile for the Jupyter TensorFlow Lite Model Maker lab environment: [Dockerfile_mediapipe](code/Dockerfile_mediapipe)
  - Conda environment specification: [environment.yml](code/environment.yml)
  - Requirements for the Streamlit service: [requirements.txt](code/requirements.txt)
  - Jupyter notebook for fine-tuning the image classification model: [custom_image_classifier_model_training.ipynb](code/custom_image_classifier_model_training.ipynb)
  - Jupyter notebook for training custom Pose Landmarker Model with MediaPipe Model Maker (!!!BROKEN!!!): [custom_pose_model_training.ipynb](code/custom_pose_model_training.ipynb)
  - Jupyter notebook for the baseline image classification workflow: [image_classification.ipynb](code/image_classification.ipynb)
  - Jupyter notebook for converting images or videos to annotated posture analytics: [human_posture_analysis.ipynb](code/human_posture_analysis.ipynb)
  - Python script for converting images or videos to annotated posture analytics: [human_posture_analysis.py](code/human_posture_analysis.py)
  - Streamlit app for pose detection and analysis: [mediapipe_pose.py](code/mediapipe_pose.py)
  - Subfolder `code/posture-keypoints-detection/` containing the code for fine-tuning and deploying the YOLO11s-pose model:
    - Jupyter notebook for fine‑tuneing a YOLO11s‑pose model on a custom, CVAT‑annotated dataset of 300 side‑view posture images to learn spinal keypoint detection for automated posture assessment.: [train.ipynb](code/posture-keypoints-detection/train.ipynb)
    - Jupyter notebook for model inference and evaluation of the fine-tuned YOLO11s-pose model: [inference.ipynb](code/posture-keypoints-detection/inference.ipynb)
    - Frontend folder for the Streamlit app: [frontend](code/posture-keypoints-detection/frontend/)
    - Backend folder for the FastAPI service: [backend](code/posture-keypoints-detection/backend/)
  - Subfolder `/code/scripts` containing helper functions for the Streamlit app: [scripts](code/scripts/*)
    - `custom_tflite_pose.py`: helper functions for loading and running inference with a custom TensorFlow Lite model for pose classification
    - `custom_tflite_image_classifier.py`: helper functions for loading and running inference with a custom TensorFlow Lite model for image classification
    - `mediapipe_api.py`: n8n API wrapper functions for MediaPipe Pose
- Documents (folder): [docs](docs/)
  - Proposal documents: [Proposal](docs/Proposal/)
  - Reflective Blog documents: [Reflective Blog](docs/Reflective Blog/)
  - Ethics documents: [Ethics](docs/Ethics/)
  - Supervision meeting diary template: [MSc_DSHSC_Supervision_research_meeting_diary_template.docx](docs/MSc_DSHSC_Supervision_research_meeting_diary_template.docx)
  - Bachelor thesis of a Serious Game in Health project: [Bachelor Thesis](docs/Bachelor Thesis/Bachelor.Thesis.pdf)
- Literature (folder): [literature](literature/)
  - `bibliography.bib`
  - `apa.csl`
<!-- markdownlint-enable MD051 -->

## Build dissertation artefacts

The dissertation artefacts can be built using Quarto. The generated PDF will be available at `Dissertation.pdf` after the build.

```bash
# Rendering to PDf
quarto render Dissertation.qmd --to pdf

# Rendering to HTML
quarto render Dissertation.qmd --to html

# Rendering to Word
quarto render Dissertation.qmd --to docx
```

## Build developing environments (Jenkins & Docker)

The repository ships with a Docker setup tailored for TensorFlow Lite Model Maker and the surrounding tooling. Set a password for the bundled Jupyter server by exporting `JUPYTER_PASSWORD` or defining it in a `.env` file next to `code/docker-compose.yml` before launching the containers. The container will abort startup if the variable is empty. Use Docker Compose to build and launch the lab environment locally (or rely on the Jenkins build that is automatically triggered on pushes to the repository):

```bash
# Stopping any running container
docker compose -f code/docker-compose.yml down

# Building container
docker compose -f code/docker-compose.yml build  #--progress plain

# Starting both services in detached mode
docker compose -f code/docker-compose.yml up -d

# Logging container output
docker compose -f code/docker-compose.yml logs -f dissertation-tflite-lab-1
```

Open a shell inside the container to retrieve the login URL, then authenticate in the browser with the password stored in `JUPYTER_PASSWORD`:

```bash
# Getting the Jupyter url with token
docker compose -f code/docker-compose.yml exec tflite-lab jupyter notebook list
```

The Streamlit pose explorer is available at [http://localhost:8501](http://localhost:8501) once the `streamlit` service is running.

> **Compatibility note:** The Docker image pins TensorFlow to 2.8.0 and constrains the scientific stack to versions compatible with `tflite-model-maker==0.4.3`. If you also need the newer `mediapipe-model-maker` pipeline, consider creating a parallel container with an updated TensorFlow stack to avoid conflicting requirements.

## Mediapipe Pose

- [MediaPipe Studio Home](https://mediapipe-studio.webapps.google.com/home)
- [MediaPipe Pose Landmarker Demo](https://mediapipe-studio.webapps.google.com/studio/demo/pose_landmarker)

Webapp demo:
[MediaPipe Pose Tracking Demo](https://viz.mediapipe.dev/demo/pose_tracking)

### TLite Model Maker for Image Classification

Training a custom image classification model using MediaPipe Model Maker.

`code/custom_image_classifier_model_training.ipynb`

Server-first workflow to commit and push trained models:

```bash
# 1) Login to the server
ssh -i ~/.ssh/id_rsa root@ssh.seriousbenentertainment.org

# 2) Open a shell in the running Jupyter container
docker exec -it dissertation-tflite-lab-1 bash

# 3) Go to the project repository mounted in the container
cd /workspace/project

# 4) Ensure git can operate in this mounted path and set identity (first time only)
git config --global --add safe.directory /workspace/project
git config user.name "DrBenjamin"
git config user.email "s2616861@ed.ac.uk"

# 5) Start a dedicated branch for generated model artifacts
git switch -c server-model-build-$(date +%Y%m%d)

# 6) Stage notebooks and newly trained model artifacts
git add code/custom_image_classifier_model_training.ipynb
git add data/models/efficientnet_lite0 data/models/efficientnet_lite2 data/models/efficientnet_lite4 data/models/mobilenet_v2

# 7) Commit
git commit -m "Add server container model build artifacts"

# 8) Push branch to GitHub
git push -u origin "$(git rev-parse --abbrev-ref HEAD)"
```

Optional one-time HTTPS push with token (if `origin` credentials are not configured):

```bash
git push "https://<GITHUB_USERNAME>:<GITHUB_TOKEN>@github.com/DrBenjamin/dissertation-movement-analysis.git" "$(git rev-parse --abbrev-ref HEAD)"
```

After push, open the PR page:

```bash
echo "https://github.com/DrBenjamin/dissertation-movement-analysis/pull/new/$(git rev-parse --abbrev-ref HEAD)"
```

The exported models can be used in the Streamlit pose detection app (`code/mediapipe_pose.py`).

### Images or videos posture analysis

Converting images and videos to annotated outputs measuring the posture on the MediaPipe Pose landmarks using this [OpenCV Tutorial](https://learnopencv.com/building-a-body-posture-analysis-system-using-mediapipe/).

Files: `code/human_posture_analysis.ipynb` and `code/human_posture_analysis.py`.

To run the Python script:

```bash
# for images
python code/human_posture_analysis.py --mode image --api-base-url http://seriousbenentertainment.org:8000 --input-video ./data/images/input.png --output-video ./data/images/output.png

# for videos
python code/human_posture_analysis.py --mode video --api-base-url http://seriousbenentertainment.org:8000 --input-video ./data/video/input.mp4 --output-video ./data/video/output.mp4

# for videos with the worst posture frame extracted as image
python code/human_posture_analysis.py --mode video --api-base-url http://seriousbenentertainment.org:8000 --input-video ./data/video/input.mp4 --output-video ./data/video/output.mp4 --output-image ./data/video/output_worst_frame.png
```

### Streamlit MediaPipe Pose App

You can now run a local Streamlit application to experiment with MediaPipe Pose on your own images.

Run the app:

```bash
python -m streamlit run code/mediapipe_pose.py
```

Features:

- Multiple image upload
- Configurable model complexity and confidence thresholds
- Optional segmentation mask blending
- Display of pixel nose coordinates and sample world landmark
- Download of annotated images as a zip archive

Planned enhancements (not yet implemented): video support, CSV export of all landmarks, comparative analytics view.

## References

All references and resources used in the project are listed below.

### Mediapipe-model-maker

[MediaPipe Model Maker – Getting Started](https://ai.google.dev/edge/mediapipe/solutions/model_maker#get_started)
[MediaPipe Model Maker – Image Classifier Customisation](https://ai.google.dev/edge/mediapipe/solutions/customization/image_classifier)

**In Colab:**
[MediaPipe Model Maker Colab Example](https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/customization/image_classifier.ipynb#scrollTo=Nbu3mnPiSvSn)

Custom TensorFlow Lite models for image classification on-device using MediaPipe Model Maker:
[DeepWiki – Custom TensorFlow Lite Models](https://deepwiki.com/search/is-it-possible-to-develop-a-ml_ee90fb1b-8cfd-4e5d-ab63-22ba7a4bc499)

### Human 3D models compatibility

[Human Mesh Recovery Survey (arXiv 2212.14474)](https://arxiv.org/pdf/2212.14474)
[PosePile Dataset](https://github.com/isarandi/PosePile)
[Pose Dataset Viewer](https://github.com/isarandi/pose-dataset-viewer)

**Demo:**

[MediaPipe Studio](https://mediapipe-studio.webapps.google.com/studio/demo/pose_landmarker)
