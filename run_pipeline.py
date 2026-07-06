"""Full pipeline entry point: camera feed -> classify pieces -> solve -> drive robot.

Captures a frame from the camera (same source as feed.py), feeds it into the
side classifier, solves the assembly, then runs the pyfirmata controller.
"""

import argparse
import os
import subprocess
import sys

import cv2
from PIL import Image

import getsides
from solve import solve

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def top_square_rect(frame):
    """Return (x, y, w, h) for the largest square at the TOP of the frame,
    horizontally centered. For a portrait iPhone feed this is the full-width
    top region; the puzzle is expected to sit there."""
    h, w = frame.shape[:2]
    side = min(h, w)
    x0 = (w - side) // 2
    return x0, 0, side, side


def crop_top_square(frame):
    x, y, w, h = top_square_rect(frame)
    return frame[y:y + h, x:x + w]


def clamp_rect(frame, rect):
    """Clamp (x, y, w, h) to frame bounds and return the cropped frame."""
    h, w = frame.shape[:2]
    x, y, rw, rh = rect
    x = max(0, min(int(x), w - 1))
    y = max(0, min(int(y), h - 1))
    rw = max(1, min(int(rw), w - x))
    rh = max(1, min(int(rh), h - y))
    return frame[y:y + rh, x:x + rw]


def draw_hud(img, lines):
    """Draw instruction text with a dark outline so it reads on any background."""
    y = 34
    for line in lines:
        cv2.putText(img, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        y += 36


def rect_from_drag(drag):
    """Convert a drag {start,end} into (x, y, w, h), or None if too small."""
    if not (drag["start"] and drag["end"]):
        return None
    x0, y0 = drag["start"]
    x1, y1 = drag["end"]
    rw, rh = abs(x1 - x0), abs(y1 - y0)
    if rw <= 5 or rh <= 5:
        return None
    return min(x0, x1), min(y0, y1), rw, rh


def select_crop(frame):
    """Interactive drag-to-select crop on a still image (used for --image)."""
    win = "Crop: drag a box | ENTER=confirm  R=reset  Q=full frame"
    drag = {"start": None, "end": None, "drawing": False}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drag["start"] = (x, y); drag["end"] = (x, y); drag["drawing"] = True
        elif event == cv2.EVENT_MOUSEMOVE and drag["drawing"]:
            drag["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drag["end"] = (x, y); drag["drawing"] = False

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    h, w = frame.shape[:2]
    cv2.resizeWindow(win, int(w * min(1.0, 1100.0 / w)), int(h * min(1.0, 1100.0 / w)))
    cv2.imshow(win, frame)
    cv2.setMouseCallback(win, on_mouse)

    rect = None
    while True:
        disp = frame.copy()
        if drag["start"] and drag["end"]:
            cv2.rectangle(disp, drag["start"], drag["end"], (0, 255, 0), 3)
        draw_hud(disp, ["Drag a box over the puzzle",
                        "ENTER=confirm  R=reset  Q=full frame"])
        cv2.imshow(win, disp)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32):
            rect = rect_from_drag(drag)
            if rect:
                break
            print("Draw a larger box first (or press Q to keep the full frame).")
        elif key == ord('r'):
            drag["start"] = drag["end"] = None
        elif key == ord('q'):
            break

    try:
        cv2.destroyWindow(win)
    except cv2.error:
        pass

    if rect is None:
        print("No region selected; using full frame.")
        return frame
    return clamp_rect(frame, rect)


def apply_crop(frame, mode):
    if mode == "top-square":
        return crop_top_square(frame)
    if mode == "select":
        return select_crop(frame)
    return frame  # "none"


def capture_frame(camera_index=1, crop="select", width=1920, height=1080):
    """Open the live feed, let the user crop, and return the final BGR frame.

    select      -> drag a box directly on the LIVE feed, ENTER to capture+crop.
    top-square  -> live overlay of the kept square, SPACE captures.
    none        -> SPACE captures the full frame.

    width/height request a capture resolution from the virtual camera. OpenCV
    otherwise defaults to ~640x480, which makes the feed look blurry.
    """
    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {camera_index}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Requested {width}x{height}, camera negotiated {actual_w}x{actual_h}")
    if actual_w < 1280:
        print("WARNING: capture resolution is low — the feed will look blurry. "
              "Check OBS Virtual Camera output resolution / OBS canvas size.")

    win = "Pipeline Feed"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    drag = {"start": None, "end": None, "drawing": False}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drag["start"] = (x, y); drag["end"] = (x, y); drag["drawing"] = True
        elif event == cv2.EVENT_MOUSEMOVE and drag["drawing"]:
            drag["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drag["end"] = (x, y); drag["drawing"] = False

    if crop == "select":
        cv2.setMouseCallback(win, on_mouse)

    print("Live feed open. See the on-screen instructions in the window.")

    captured = None
    rect = None
    sized = False
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break

            if not sized:
                h, w = frame.shape[:2]
                scale = min(1.0, 1100.0 / w)
                cv2.resizeWindow(win, int(w * scale), int(h * scale))
                sized = True

            disp = frame.copy()
            if crop == "select":
                if drag["start"] and drag["end"]:
                    cv2.rectangle(disp, drag["start"], drag["end"], (0, 255, 0), 3)
                draw_hud(disp, ["SELECT: drag a box over the puzzle",
                                "ENTER=use box  F=full frame  R=reset  Q=abort"])
            elif crop == "top-square":
                x, y, sw, sh = top_square_rect(frame)
                cv2.rectangle(disp, (x, y), (x + sw, y + sh), (0, 255, 0), 3)
                draw_hud(disp, ["TOP-SQUARE crop", "SPACE=capture  Q=abort"])
            else:
                draw_hud(disp, ["FULL frame", "SPACE=capture  Q=abort"])

            cv2.imshow(win, disp)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('q'):
                break

            if crop == "select":
                if key in (13, 32):  # ENTER / SPACE = use the drawn box
                    rect = rect_from_drag(drag)
                    if rect:
                        captured = frame.copy()
                        break
                    print("Draw a larger box first, or press F for the full frame.")
                elif key == ord('f'):
                    captured = frame.copy(); rect = None
                    break
                elif key == ord('r'):
                    drag["start"] = drag["end"] = None
            else:
                if key == 32:  # SPACE
                    captured = frame.copy()
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if captured is None:
        raise RuntimeError("No frame captured")

    if crop == "select":
        return captured if rect is None else clamp_rect(captured, rect)
    if crop == "top-square":
        return crop_top_square(captured)
    return captured


def list_cameras(max_index=5):
    """Probe camera indices and print each one's negotiated resolution.

    Run this once after connecting the iPhone (Continuity Camera) to find which
    index it lands on, then pass that to --camera-index.
    """
    print("Probing camera indices (CAP_AVFOUNDATION)...")
    found = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"  index {i}: OPEN   {w}x{h}")
                found.append(i)
            else:
                print(f"  index {i}: opened but returned no frame")
        else:
            print(f"  index {i}: not available")
        cap.release()
    if not found:
        print("No working cameras found. Is the iPhone connected/awake and the app running?")
    return found


def main():
    parser = argparse.ArgumentParser(
        description="Run the full puzzle pipeline: feed -> classify -> solve -> robot."
    )
    parser.add_argument("--camera-index", type=int, default=1,
                        help="OpenCV camera index for the feed (default: 1).")
    parser.add_argument("--image", type=str, default=None,
                        help="Skip the camera and use an existing image file as the feed.")
    parser.add_argument("--no-robot", action="store_true",
                        help="Stop after solving; do not run the pyfirmata controller.")
    parser.add_argument("--crop", choices=["select", "top-square", "none"],
                        default="select",
                        help="Crop the feed before classifying. 'select' (default) lets you "
                             "drag a box on the captured frame; 'top-square' keeps the square "
                             "top region of the (rectangular) iPhone feed; 'none' keeps the "
                             "full frame.")
    parser.add_argument("--width", type=int, default=1920,
                        help="Requested capture width (default: 1920). Higher = sharper feed.")
    parser.add_argument("--height", type=int, default=1080,
                        help="Requested capture height (default: 1080).")
    args = parser.parse_args()

    # getsides/solve use relative paths (pieces_data.json, results/, images/).
    os.chdir(BASE_DIR)

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise RuntimeError(f"Failed to read image file: {args.image}")
        print(f"Using feed image: {args.image}")
        frame = apply_crop(frame, args.crop)
    else:
        # capture_frame applies the crop interactively (live drag for 'select').
        frame = capture_frame(args.camera_index, crop=args.crop,
                              width=args.width, height=args.height)

    print(f"Crop mode '{args.crop}' -> feed size {frame.shape[1]}x{frame.shape[0]}")
    if min(frame.shape[:2]) < 300:
        print("WARNING: cropped region is small (<300px) — the puzzle area is low-res. "
              "Make the iPhone source fill more of the OBS canvas, or crop a larger box.")
    feed_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    feed_path = os.path.join(BASE_DIR, "images", "captured_feed.png")
    os.makedirs(os.path.dirname(feed_path), exist_ok=True)
    feed_image.save(feed_path)
    print(f"Saved feed image to {feed_path}")

    # Inject the captured frame as the image the classifier reads (getsides
    # references the module-global `image`, normally loaded by utils.py).
    getsides.image = feed_image

    # Classify pieces and solve. refresh_sides=True forces getsides to run on
    # the new frame and regenerate pieces_data.json before solving.
    solve(refresh_sides=True)

    if args.no_robot:
        print("Skipping robot step (--no-robot).")
        return

    print("Running controller (pyfirmata)...")
    subprocess.run([sys.executable, os.path.join(BASE_DIR, "controller.py")], check=True)


if __name__ == "__main__":
    main()
