import cv2
import numpy as np
import subprocess
import os
from scipy.signal import butter, lfilter, welch
from tqdm import tqdm

# -----------------------------
# STEP 1: Convert video
# -----------------------------
def convert_video(input_path, output_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video file not found: {input_path}")
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vcodec", "libx264",
        "-preset", "fast",
        "-crf", "18",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path

# -----------------------------
# STEP 2: Stabilize video
# -----------------------------
def stabilize_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    transforms = []

    prev_gray = None

    for i in tqdm(range(n_frames), desc="Stabilizing"):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            prev_gray = gray
            transforms.append((0,0))
            continue

        prev_pts = cv2.goodFeaturesToTrack(prev_gray, 200, 0.01, 30)
        
        if prev_pts is None or len(prev_pts) == 0:
            transforms.append((0, 0))
            prev_gray = gray
            continue

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)

        if status is None or not np.any(status == 1):
            transforms.append((0, 0))
            prev_gray = gray
            continue

        valid_prev = prev_pts[status==1]
        valid_curr = curr_pts[status==1]

        dx = np.mean(valid_curr[:,0] - valid_prev[:,0])
        dy = np.mean(valid_curr[:,1] - valid_prev[:,1])

        transforms.append((dx, dy))
        prev_gray = gray

    cap.release()

    if not transforms:
        raise ValueError("No frames were successfully read from the video for stabilization.")

    # Smooth motion
    transforms = np.array(transforms)
    smoothed = np.convolve(transforms[:,0], np.ones(30)/30, mode='same')
    smoothed_y = np.convolve(transforms[:,1], np.ones(30)/30, mode='same')

    cap = cv2.VideoCapture(input_path)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))

    for i in tqdm(range(len(transforms)), desc="Applying stabilization"):
        ret, frame = cap.read()
        if not ret:
            break

        dx = transforms[i][0] - smoothed[i]
        dy = transforms[i][1] - smoothed_y[i]

        M = np.float32([[1,0,-dx],[0,1,-dy]])
        stabilized = cv2.warpAffine(frame, M, (w,h))

        out.write(stabilized)

    cap.release()
    out.release()
    return output_path

# -----------------------------
# STEP 3: Bandpass filter
# -----------------------------
def butter_bandpass(lowcut, highcut, fs, order=1):
    nyq = 0.5 * fs
    return butter(order, [lowcut/nyq, highcut/nyq], btype='band')

def bandpass(data, lowcut, highcut, fs):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return lfilter(b, a, data, axis=0)

# -----------------------------
# STEP 4: EVM
# -----------------------------
def evm(video_path, output_path, alpha=40, low=0.5, high=3.0, chunk_size=10):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video file not found for EVM: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))

    b, a = butter_bandpass(low, high, fps)
    
    from scipy.signal import lfilter
    zi = np.zeros((max(len(a), len(b)) - 1, h, w, 3), dtype=np.float32)
    motion_list = []

    while True:
        chunk = []
        for _ in range(chunk_size):
            ret, frame = cap.read()
            if not ret:
                break
            chunk.append(frame.astype(np.float32)/255.0)

        if not chunk:
            break

        frames = np.array(chunk, dtype=np.float32)

        filtered, zi = lfilter(b, a, frames, axis=0, zi=zi)
        amplified = frames + alpha * filtered

        amplified = np.clip(amplified, 0, 1)
        
        # Compute global motion signal for the chunk
        gray_frames = np.mean(amplified, axis=3)
        chunk_motion = np.std(gray_frames, axis=(1,2))
        motion_list.extend(chunk_motion)

        for f in amplified:
            out.write((f*255).astype(np.uint8))

    cap.release()
    out.release()
    return np.array(motion_list), fps

# -----------------------------
# STEP 5: Earthquake detection
# -----------------------------
def detect_earthquake(motion, fps):
    # Frequency analysis
    freqs, power = welch(motion, fs=fps)

    # Earthquake band
    mask = (freqs >= 0.5) & (freqs <= 5)
    quake_power = np.sum(power[mask])
    total_power = np.sum(power)

    ratio = quake_power / total_power

    print(f"\nEarthquake frequency energy ratio: {ratio:.3f}")

    # Heuristic decision
    if ratio > 0.6:
        print("⚠️ HIGH likelihood of earthquake-like vibration")
    elif ratio > 0.3:
        print("⚠️ POSSIBLE earthquake-like motion")
    else:
        print("❌ Likely NOT an earthquake (camera or local movement)")

# -----------------------------
# MAIN PIPELINE
# -----------------------------
if __name__ == "__main__":

    input_file = "BalangirQuake2026.mp4"

    converted = "converted.mp4"
    stabilized = "stabilized.mp4"
    output = "amplified.mp4"

    print("Step 1: Converting video...")
    convert_video(input_file, converted)

    print("Step 2: Stabilizing video...")
    stabilize_video(converted, stabilized)

    print("Step 3: Applying EVM...")
    motion, fps = evm(stabilized, output)

    print("Step 4: Detecting earthquake...")
    detect_earthquake(motion, fps)

    print("\nDone! Output saved as:", output)