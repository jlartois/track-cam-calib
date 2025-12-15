from tqdm import tqdm
import numpy as np
np.set_printoptions(suppress=True)
from loader import *
from calib import *


############# FILL IN PARAMETERS HERE ###################
N = 25                   # every N frames, keypoints are reset (`k` in book)
json_path = "camera_poses.json"
video_path = "video.mkv"
intrinsics = {
    "Resolution": [1920, 1080],
    "Focal": [1371.0, 1371.0],
    "Principal_Point": [1920 * 0.5, 1080 * 0.5],
    "K": np.array([[1371.0, 0, 1920 * 0.5],
                   [0, 1371.0, 1080 * 0.5],
                   [0, 0, 1]], dtype=np.float32)
}
feature_params = dict(maxCorners = 1000, qualityLevel = 0.1, minDistance = 7) # qualityLevel is `q` in book
#########################################################

extrinsics = load_extrinsics(json_path)
frames = load_video(video_path)
nr_frames = len(frames)
nr_frames = ((nr_frames - 1) // N) * N

all_estimated_extrinsics = {}
all_pos_errors = []
all_rot_errors = []
all_nr_outliers = []
all_nr_keypoints = []
all_reproj_errors = []

# every N-th frame
for i in tqdm(range(0, nr_frames-1, N)):
    keypoints = find_good_keypoints(frames[i], i, feature_params)
    
    keypoints_per_frame = track_keypoints_over_N_frames(keypoints, frames[i:i+N+1], N)
    
    points = triangulate_keypoints(keypoints_per_frame[0], keypoints_per_frame[-1], extrinsics[i], extrinsics[i+N], intrinsics)
    
    reproj_error0, is_outlier0 = calculate_reprojection_error(keypoints_per_frame[ 0], points, extrinsics[i], intrinsics)
    reproj_error1, is_outlier1 = calculate_reprojection_error(keypoints_per_frame[-1], points, extrinsics[i+N], intrinsics)
    all_reproj_errors.extend([reproj_error0, reproj_error1])
    
    # remove (key)points with a high reprojection error
    is_outlier = np.logical_or(is_outlier0, is_outlier1)
    nr_outliers = np.sum(is_outlier).item()
    all_nr_outliers.append(nr_outliers + keypoints.shape[0] - keypoints_per_frame.shape[1])
    
    keypoints_per_frame = keypoints_per_frame[:,~is_outlier,:]
    points = points[:,~is_outlier]
    all_nr_keypoints.append(keypoints_per_frame.shape[1])
    
    ######
    for j in range(0, N):
        vis = frames[i+j].copy()
        for k in keypoints_per_frame[j]:
            vis = cv2.circle(vis, (int(k[0]), int(k[1])), 4, (0,255,0), -1)
            
        nr = len(os.listdir("deleteme"))
        cv2.imwrite("deleteme/{:04d}.png".format(nr), vis)
    ####
    
    estimated_extrinsics = estimate_intermediate_poses(keypoints_per_frame, points, N, intrinsics, extrinsics[i]["model_euler"])
    for j,extr in enumerate(estimated_extrinsics):
    
        curr_frame = i+j+1
        all_estimated_extrinsics[curr_frame] = extr
        
        # calculate the distance between the estimated and ground truth position
        pos = extr["model"][:3, 3:]
        pos_gr = extrinsics[curr_frame]["model"][:3, 3:]
        pos_error = np.linalg.norm(np.abs(pos - pos_gr), axis=0)
        all_pos_errors.append(pos_error)
        
        # calculate the distance between the estimated and ground truth rotation (in degrees)
        rot = extr["model_euler"]
        rot_gr = extrinsics[curr_frame]["model_euler"]
        rot_error = np.linalg.norm(np.abs(rot - rot_gr), axis=0)
        rot_error = np.rad2deg(rot_error)
        all_rot_errors.append(rot_error)
    
print("Summary:")
print("k = {}, nr S images = {}, quality_level = {}".format(N, len(all_nr_keypoints)+1, feature_params['qualityLevel']))
print("reprojection error of S: mean = {:.2f} px, stdev = {:.3f} px".format(np.mean(all_reproj_errors), np.std(all_reproj_errors)))
print("position error: mean = {:.4f} m, stdev = {:.5f} m".format(np.mean(all_pos_errors), np.std(all_pos_errors)))
print("rotation error: mean = {:.4f} degrees, stdev = {:.5f} degrees".format(np.mean(all_rot_errors), np.std(all_rot_errors)))
print("#keypoints: {}".format(np.sum(all_nr_keypoints)))
print("#outlier keypoints removed: {}".format(np.sum(all_nr_outliers)))