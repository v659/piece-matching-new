# Piece Matching v1.0

A research-style Python project that extracts jigsaw piece boundaries from an image and attempts to assemble the puzzle by matching complementary side profiles.

## Current Status

This project is experimental and currently optimized for the author’s sample images. Solver quality can vary significantly across inputs.

## Repository Layout

- `getsides.py`: Detects pieces, extracts side segments, classifies side types, and writes piece metadata.
- `solve.py`: Matches sides, estimates piece poses, builds an assembled layout, and writes movement plans.
- `side.py`: `Side` model and geometry helpers for individual piece sides.
- `utils.py`: Image preprocessing, contour extraction, corner detection, and shape utilities.
- `images/`: Sample puzzle images.
- `results/`: Optional output folder (not currently used by scripts by default).

## Requirements

- Python 3.12+ (project appears to have been run with 3.12/3.14)
- macOS/Linux/Windows

Install dependencies:

```bash
pip install numpy scipy matplotlib opencv-python pillow shapely scikit-learn rdp pyparsing six
```

## Quick Start

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Configure input image path in `utils.py`.

The pipeline reads a hardcoded path:

```python
image_path = resource_path("puzzle-final3.png")
```

Update it to a real file path in your workspace, for example:

```python
image_path = resource_path("images/puzzle-final3.png")
```

3. Extract sides:

```bash
python getsides.py
```

4. Solve puzzle layout:

```bash
python solve.py
```

Optional: force fresh side extraction before solving:

```bash
python solve.py --refresh-sides
```

## Generated Files

Running `getsides.py` writes:

- `binarized_image.png`
- `piece_shapes.png`
- `side_segments.png`
- `classified` (saved without extension in current code)
- `pieces_data.json`
- `types_data.json`

Running `solve.py` writes:

- `solved_assembly.png`
- `assembly_steps.json`

`solve.py` also attempts to display an interactive animation window.

## Solver Notes

- Side matching is based on profile similarity features and assignment across outward/inward side pairs.
- Placement starts from an anchor piece, then grows using geometric alignment and overlap checks.
- Remaining pieces are attached with relaxed thresholds when confident matches are not found in the first pass.
- Output includes per-piece rotation/translation instructions in `assembly_steps.json`.

## Known Limitations

- Input image selection is currently hardcoded in `utils.py`.
- Side classification and matching heuristics are sensitive to image quality, thresholding, and erosion settings.
- The project does not guarantee a correct full solve on arbitrary puzzles.

## License

MIT. See Liscence file for details.
