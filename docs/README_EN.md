# VLA-Arena Documentation Table of Contents (English)

This document provides a comprehensive table of contents for all VLA-Arena documentation files.

## 📚 Complete Documentation Overview

### 1. Data Collection Guide
**File:** `data_collection.md`

A comprehensive guide for collecting demonstration data in custom scenes and converting data formats.

#### Table of Contents:
1. [Collect Demonstration Data](#1-collect-demonstration-data)
   - Interactive simulation environment setup
   - Keyboard controls for robotic arm manipulation
   - Data collection process and best practices
2. [Convert Data Format](#2-convert-data-format)
   - Converting demonstration data to training format
   - Image generation through trajectory replay
   - Dataset creation process
3. [Regenerate Dataset](#3-regenerate-dataset)
   - Filtering noop actions for trajectory continuity
   - Dataset optimization and validation
   - Quality assurance procedures
4. [Convert Dataset to RLDS Format](#4-convert-dataset-to-rlds-format)
   - RLDS format conversion
   - Dataset standardization
5. [Convert RLDS Dataset to LeRobot Format](#5-convert-rlds-dataset-to-lerobot-format)
   - LeRobot format conversion
   - Compatibility handling

---

### 2. Scene Construction Guide
**File:** `scene_construction.md`

Detailed guide for building custom task scenarios using BDDL (Behavior Domain Definition Language).

#### Table of Contents:
1. [BDDL File Structure](#1-bddl-file-structure)
   - Basic structure definition
   - Domain and problem definition
   - Language instruction specification
2. [Region Definition](#region-definition)
   - Spatial scope definition
   - Region parameters and configuration
3. [Object Definition](#object-definition)
   - Fixtures (static objects)
   - Manipulable objects
   - Objects of interest
   - Moving objects with motion types
4. [State Definition](#state-definition)
   - Initial state configuration
   - Goal state definition
   - Supported state predicates
5. [Image Effect Settings](#image-effect-settings)
   - Rendering effect configuration
   - Visual enhancement options
6. [Cost Constraints](#cost-constraints)
   - Penalty condition definition
   - Supported cost predicates
7. [Visualize BDDL File](#2-visualize-bddl-file)
   - Scene visualization process
   - Video generation workflow
8. [Assets](#3-assets)
   - Ready-made assets
   - Custom asset preparation
   - Asset registration process

---

### 3. Model Fine-tuning and Evaluation Guide
**File:** `finetuning_and_evaluation.md`

Comprehensive guide for fine-tuning and evaluating VLA models using VLA-Arena generated datasets. Supports OpenVLA, OpenVLA-OFT, Openpi, UniVLA, SmolVLA, and other models.

#### Table of Contents:
1. [General Models (OpenVLA, OpenVLA-OFT, UniVLA, SmolVLA)](#general-models)
   - Dependency installation
   - Model fine-tuning
   - Model evaluation
2. [Openpi Model](#openpi)
   - Environment setup (using uv)
   - Training configuration and execution
   - Policy server startup
   - Model evaluation
3. [Configuration File Notes](#configuration-file-notes)
   - Dataset path configuration
   - Model parameter settings
   - Training hyperparameter configuration

---

## 📁 Directory Structure

```
docs/
├── data_collection.md                    # Data collection guide (English)
├── data_collection_zh.md                 # Data collection guide (Chinese)
├── scene_construction.md                 # Scene construction guide (English)
├── scene_construction_zh.md              # Scene construction guide (Chinese)
├── finetuning_and_evaluation.md         # Model fine-tuning and evaluation guide (English)
├── finetuning_and_evaluation_zh.md      # Model fine-tuning and evaluation guide (Chinese)
├── README_EN.md                          # Documentation table of contents (English)
├── README_ZH.md                          # Documentation table of contents (Chinese)
└── image/                                # Documentation images and GIFs
```

---

## 🚀 Getting Started Workflow

### 1. Scene Construction
1. Read `scene_construction.md` for BDDL file structure
2. Define your task scenarios using BDDL syntax
3. Use `scripts/visualize_bddl.py` to preview scenes

### 2. Data Collection
1. Follow `data_collection.md` for demonstration collection
2. Use `scripts/collect_demonstration.py` for interactive data collection
3. Convert data format using `scripts/group_create_dataset.py`

### 3. Model Training and Evaluation
1. Follow `finetuning_and_evaluation.md` to install model dependencies
2. Use `vla-arena train` command for model fine-tuning
3. Configure training parameters according to your needs
4. Use `vla-arena eval` command to evaluate model performance
5. Monitor training progress through WandB
6. Analyze results and iterate on model improvements
