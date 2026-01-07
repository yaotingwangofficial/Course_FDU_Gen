# Course_FDU_Gen

```markdown
# Canvas Agent - Image Generation from Bounding Box Sequences

A sophisticated image generation system that creates images based on object-bounding box sequences, featuring an iterative judge-and-rebuild mechanism for quality assurance.

## Overview

This project implements an intelligent image generation agent that:
- Generates images from object-bounding box sequences
- Uses vision-language models to judge image quality
- Iteratively refines generation instructions until requirements are met
- Supports optional image editing for precise object positioning

## Architecture

The system consists of two main components:

1. **Image Generation** (`image_generation`): Generates initial images with iterative quality judgment
2. **Image Editing** (`image_edit`): Fine-tunes object positions using IOU-based detection and correction

## Project Structure

```
canavs_agent/
├── run_v6.py              # Main entry point and ImageGenerationAgentV6 class
├── helper.py              # Utility functions for bbox sequence processing
├── prompt.py              # Prompt templates for LLM interactions
├── think_tools/           # Vision-language models for reasoning
│   ├── qwen3vl.py        # Qwen3-VL model wrapper
│   └── qwen3.py          # Qwen3 text model wrapper
└── draw_tools/            # Image generation and editing models
    ├── qwen_image.py      # Qwen Image generation pipeline
    └── qwen_image_edit.py # Qwen Image Edit Plus pipeline
```

## Core Components

### 1. ImageGenerationAgentV6 (`run_v6.py`)

Main agent class with the following key methods:

- **`image_generation()`**: Core generation pipeline
  - Generates instruction from bbox sequence
  - Creates image from instruction
  - Judges quality and rebuilds if needed (up to 3 attempts)

- **`judge_and_rebuild()`**: Quality assessment
  - Uses Qwen3VL to evaluate generated images
  - Determines if image meets requirements
  - Generates new instructions if correction needed

- **`image_edit()`**: Optional refinement (IOU-based)
  - Detects object bboxes in generated image
  - Calculates IOU with target bboxes
  - Generates correction suggestions
  - Iteratively edits until IOU threshold met

### 2. Helper Functions (`helper.py`)

- `build_object_bbox_sequence()`: Constructs object-bbox sequence from lists
- `format_object_bbox_sequence()`: Formats sequence for LLM input

### 3. Prompt Templates (`prompt.py`)

Pre-defined prompts for:
- Initial instruction generation
- Image quality judgment
- Object bbox detection
- Correction suggestion generation

### 4. Think Tools (`think_tools/`)

Vision-language models for reasoning:
- **Qwen3VL**: Multi-modal model for image understanding and instruction generation
- **Qwen3**: Text-only model (optional)

### 5. Draw Tools (`draw_tools/`)

Image generation and editing pipelines:
- **QwenImage**: Text-to-image generation
- **QwenImageEditPlus**: Image editing with instruction

## Workflow

### Image Generation Flow

```
Input: bbox_sequence
  ↓
Generate instruction from bbox sequence
  ↓
Generate image from instruction
  ↓
Judge image quality
  ↓
[If not satisfied] → Generate new instruction → Loop
  ↓
[If satisfied] → Return final image
```

### Image Edit Flow (Optional)

```
Input: Generated image + target bbox_sequence
  ↓
Detect object bboxes in image
  ↓
Calculate IOU with target bboxes
  ↓
[If IOU < threshold] → Generate correction suggestions
  ↓
Edit image with suggestions
  ↓
[Iterate until all IOU >= threshold]
```

## Usage

### Basic Usage

```python
from run_v6 import ImageGenerationAgentV6

# Initialize agent
agent = ImageGenerationAgentV6(max_retries=3)

# Define bbox sequence
bbox_sequence = [
    {"chair": [0.3, 0.5, 0.2, 0.3]},
    {"table": [0.7, 0.5, 0.25, 0.2]}
]

# Generate image
success, image, instruction = agent.image_generation(
    bbox_sequence,
    image_w=1024,
    image_h=1024
)
```



### Main Script

```bash
python run_v6.py
```

The main script processes a JSON file containing image data and generates images for each sample.

## Configuration

Key parameters in `ImageGenerationAgentV6`:

- `max_generation_attempts`: Maximum attempts for image generation (default: 3)
- `max_edit_iterations`: Maximum iterations for image editing (default: 3)
- `iou_threshold`: IOU threshold for bbox matching (default: 0.6)

Device assignments (configurable in `__init__`):
- `qwen_image_edit`: NPU device for image editing
- `qwen3vl`: NPU device for vision-language model
- `image_draw`: NPU device for image generation

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Diffusers
- PIL (Pillow)
- Qwen models (Qwen3-VL, Qwen Image, Qwen Image Edit Plus)

## Model Paths

Default model paths (configurable):
- Qwen3-VL: `/dev/shm/Qwen3-VL-8B-Instruct`
- Qwen Image: `/root/Qwen/qwen_image`
- Qwen Image Edit Plus: `/dev/shm/Qwen-Image-Edit-2511`

