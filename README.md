# Piece Matching Project

## Overview

This project is designed to analyze and solve jigsaw puzzles from images. It processes puzzle piece images, extracts and classifies their sides (flat, inward, or outward), and attempts to assemble the puzzle by matching compatible sides using geometric transformations, similarity scoring, and a BFS-based placement algorithm.
This project is not completed, so results will not be good. Please do not ask for any other puzzles support or for PR's right now. Thankyou.
The main components include:
- Image processing to detect and binarize puzzle pieces.
- Side extraction and classification.
- Matching sides based on shape similarity using techniques like Hausdorff distance and LSH clustering.
- Puzzle assembly with transformations to align matching sides.

Key files in the project:
- `getsides.py`: Extracts sides from the input image and generates visualizations and JSON data.
- `solve.py`: Loads extracted data and solves the puzzle by matching and placing pieces.
- `utils.py`: Utility functions for image processing, geometry, and matching.
- `side.py`: Defines the `Side` class for representing and manipulating puzzle sides.
- Data files: `pieces_data.json` (piece and side data), `types_data.json` (side types), and various PNG outputs for visualizations.

## Installation

1. Ensure you have Python 3.14.0 installed.
2. Set up a virtual environment (virtualenv is configured for this project):
   ```
   virtualenv .venv
   source .venv/bin/activate  # On macOS/Linux
   .venv\Scripts\activate     # On Windows
   ```
3. Install the required packages:
   ```
   pip install matplotlib numpy opencv-python pillow pyparsing scikit-learn scipy
   ```
   Note: The project uses these packages; do not use other package managers unless specified.

## Usage

1. Place your puzzle image in the project root (e.g., `IMG_9868.jpg`).
2. Run side extraction:
   ```
   python getsides.py
   ```
   This generates:
   - `binarized_image.png`: Binarized version of the input.
   - `piece_shapes.png`: Visualized piece shapes.
   - `side_segments.png`: Detected sides with labels.
   - `classified.png`: Rotated sides with classifications.
   - `pieces_data.json`: Detailed side data for each piece.
   - `types_data.json`: Side type classifications.

3. Solve the puzzle:
   ```
   python solve.py
   ```
   This loads the JSON data, computes matches, and assembles the puzzle, printing placement details and connections.

## How It Works

1. **Image Processing (`getsides.py`)**:
   - Binarizes the input image and detects blobs (pieces).
   - Extracts edges and simplifies them using the Ramer-Douglas-Peucker (RDP) algorithm.
   - Detects corners and segments sides.
   - Classifies sides as flat, inward (trough), or outward (bulge).
   - Normalizes and rotates sides for comparison.

2. **Side Matching and Puzzle Solving (`solve.py`)**:
   - Loads piece data and uses Locality-Sensitive Hashing (LSH) with Gaussian Random Projection for fuzzy clustering of similar shapes.
   - Computes candidate matches between bulges and troughs using Hausdorff distance.
   - Uses BFS to place pieces starting from an anchor, applying rotations and translations to align matches.
   - Handles unplaced pieces by offsetting them around the assembly.

3. **Utilities and Side Class**:
   - `utils.py` provides functions for binarization, blob detection, geometry operations, normalization, and scoring.
   - `side.py` encapsulates side properties like points, index, length, angle, type, and normalization.

## Dependencies

- matplotlib
- numpy
- opencv-python (cv2)
- pillow (PIL)
- pyparsing
- scikit-learn
- scipy
- six
- rdp (for polygon simplification)

These are installed via pip in the virtualenv.

## License

See the MIT LISCENCE file for details.

