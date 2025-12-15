import numpy as np
import cv2
from math import asin, cos, atan2, pi
import os

def find_good_keypoints(image, frame_nr, feature_params):
    """
    Find keypoints in an image. 
    Returns a numpy array, np.float32, shape (nr_keypoints, 1, 2) 
    with each row containing the [column, row] of a keypoint.
    """
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    points = cv2.goodFeaturesToTrack(image_gray, **feature_params)
    
    if points is None:
        raise AssertionError("Could not find any keypoints in frame {}".format(frame_nr))
    
    return points  # shape (nr_keypoints, 1, 2)
    
def track_keypoints_over_N_frames(keypoints, frames, N, visualize=False):
    """
    Track the keypoints of frames[0] over the  next N images frames[1:], throwing away the tracks that get lost.
    Returns the keypoints for all N+1 frames, so:
    a numpy array, np.float32, shape (N+1, nr good tracks, 2) 
    with the last dimension containing the [column, row] of a keypoint.
    """
    keypoints_per_frame = np.full((N+1, keypoints.shape[0],2), -1, dtype=np.float32)
    prev_frame_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    kp0 = keypoints
    keypoints_per_frame[0,:,:] = kp0[:,0,:].copy()
    for j in range(1, N+1):
        curr_frame_gray = cv2.cvtColor(frames[j], cv2.COLOR_BGR2GRAY)
        
        kp1, st, err = cv2.calcOpticalFlowPyrLK(prev_frame_gray, curr_frame_gray, kp0, None)
        kp0_reverse, st, err = cv2.calcOpticalFlowPyrLK(curr_frame_gray, prev_frame_gray, kp1, None)
        
        # delete keypoints for which tracking was lost
        diff = abs(kp0-kp0_reverse).squeeze().max(-1)
        is_good = diff < 1 
        
        kp0 = kp1[is_good,:,:]
        keypoints_per_frame = keypoints_per_frame[:,is_good,:]
        keypoints_per_frame[j,:,:] = kp0[:,0,:].copy()
        prev_frame_gray = curr_frame_gray
        
        if visualize:
            vis = frames[j].copy()
            for k in kp1.squeeze():
                vis = cv2.circle(vis, (int(k[0]), int(k[1])), 4, (0,255,0), -1)
            cv2.imshow("Keypoints", vis)
            cv2.waitKey(30)
    
    cv2.destroyAllWindows()
    return keypoints_per_frame
    
def pixel_to_ray(col_row, model, intrinsics):
    """
    Convert a (M,2) numpy array of [column, row] to an (3,M) array of rays in world space.
    """
    M = col_row.shape[0]
    
    # unproject to Z-depth 1 using intrinsics
    ray = col_row.copy()
    ray[:,0] = (ray[:,0] - intrinsics["Principal_Point"][0]) / intrinsics["Focal"][0]
    ray[:,1] = -(ray[:,1] - intrinsics["Principal_Point"][1]) / intrinsics["Focal"][1]
    ray = np.hstack([ray, -np.ones((M, 1)), np.ones((M,1))])
    
    # to world space
    ray = model @ np.transpose(ray)
    
    return ray[:3,:] # (3,M)
    
def point_to_pixel(points, view, intrinsics):
    """
    Convert a (3,M) numpy array of points to a (M,2) array of [column,row] pixels.
    """
    points = view @ np.vstack([points, np.ones((1,points.shape[1]))])
    points[0,:] = -points[0,:] / points[2,:] * intrinsics["Focal"][0] + intrinsics["Principal_Point"][0]
    points[1,:] =  points[1,:] / points[2,:] * intrinsics["Focal"][1] + intrinsics["Principal_Point"][1]
    return np.transpose(points[:2,:])
    
    
def closest_point_between_2_rays(o1, p1, o2, p2):
    """
    A ray here is defined by two points it passes through, for example o1 and p1.
    Returns the 3D point closest to both rays
    """
    
    d1 = p1 - o1
    d2 = p2 - o2
    n = np.cross(d1, d2, axis=0)
    n1 = np.cross(d1, n, axis=0)
    n2 = np.cross(d2, n, axis=0)
    
    # achieve dot product along axis 0 through an elementwise multiplication and sum
    s1 = np.sum((o2-o1) * n2, axis=0) / np.sum(d1 * n2, axis=0) 
    s2 = np.sum((o1-o2) * n1, axis=0) / np.sum(d2 * n1, axis=0) 
    
    avg_points = 0.5 * (o1+s1*d1 + o2+s2*d2)
    #avg_points = o1+s1*d1
    #avg_points = o2+s2*d2
    return avg_points
    
def triangulate_keypoints(kp0, kp1, extrinsics0, extrinsics1, intrinsics):
    """
    Given the [column,row] of a keypoint in image0 and image1, and the intrinsics and extrinsics,
    triangulate the 3D point corresponding to that keypoint.
    Returns a (3,M) numpy array.
    """
    rays0 = pixel_to_ray(kp0, extrinsics0["model"], intrinsics)
    rays1 = pixel_to_ray(kp1, extrinsics1["model"], intrinsics)
    
    cam_pos0 = extrinsics0["model"][:3,3].reshape((3,1))
    cam_pos1 = extrinsics1["model"][:3,3].reshape((3,1))
    
    points = closest_point_between_2_rays(cam_pos0, rays0, cam_pos1, rays1)
    return points
    
def calculate_reprojection_error(keypoints, points, extrinsics, intrinsics):
    """
    Returns a (M,) numpy array mask, which is True for keypoints with a high reprojection error
    """
    col_row = point_to_pixel(points, extrinsics["view"], intrinsics)
    
    # take the mean abs difference of the projected points and the original pixel positions
    error = np.abs(col_row - keypoints)
    error = np.linalg.norm(error, axis=1)  # shape (M,)
    mean_error = np.mean(error)
    
    # detect outliers with a high reprojection error
    is_outlier = error > 3
    return mean_error, is_outlier
    
def closest(values, target):
    dist = [abs(v-target) for v in values]
    closest_idx = dist.index(min(dist))
    return values[closest_idx]

def rotation_matrix_to_euler_angles(R, reference_euler):
    assert(abs(abs(R[2][0]) - 1) > 0.0001)
    ry_s = [-asin(R[2][0]),  pi +asin(R[2][0])] # 2 solutions
    ry = closest(ry_s, reference_euler[1])
    cos_ry = cos(ry)
    assert(abs(cos_ry) > 0.0001)
    rx = atan2(R[2][1]/cos_ry, R[2][2]/cos_ry)
    rz = atan2(R[1][0]/cos_ry, R[0][0]/cos_ry)
    return np.array([rx, ry, rz])
    
def estimate_intermediate_poses(keypoints_per_frame, points, N, intrinsics, reference_euler):
    camera_matrix = intrinsics["K"]
    dist_coeffs = np.array([0, 0, 0, 0], dtype=np.float32)
    estimated_extrinsics = []
    flip = np.diag([1, -1, -1, 1]).astype(np.float32) # to flip between Blender's and OpenCV's axial system
    object_points = np.transpose(points).astype(np.float32)
    object_points[:,1] *= -1   # from Blender to OpenCV axial system
    object_points[:,2] *= -1
    
    for j in range(1,N):
        # PnP to get pose of current camera
        image_points = np.ascontiguousarray(keypoints_per_frame[j])
        success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            raise AssertionError("cv2.solvePnP failed")
            
        # convert to model and view matrices
        view_rot_3x3, __ = cv2.Rodrigues(rvec)
        view = np.identity(4, dtype=np.float32)
        view[:3,:3] = view_rot_3x3
        view[:3, 3:] = tvec
        view = flip @ view @ flip  # to Blender axial system
        model = np.linalg.inv(view)
        
        # now also try to find the euler_angles for view and model
        estimated_euler_angles_view = rotation_matrix_to_euler_angles(view[:3,:3], -reference_euler)
        estimated_euler_angles_model = rotation_matrix_to_euler_angles(model[:3,:3], reference_euler)
        
        estimated_extrinsics.append({
            'view': view,
            'model': model,
            'view_euler': estimated_euler_angles_view,
            'model_euler': estimated_euler_angles_model
        })
    return estimated_extrinsics