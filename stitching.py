'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function stitch_background() and panorama().
3. If you want to show an image for debugging, please use show_image() function in util.py. 
4. Please do NOT save any intermediate files in your final submission.
'''
import torch
import kornia as K
from typing import Dict
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

# ------------------------------------ Helper Functions ------------------------------------ #

#Setting fixed seed for reproducibility and deterministic behavior. To avoid randomness.
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)

#Converting the image to grayscale using Kornia
def to_gray(img):
    return K.color.rgb_to_grayscale(img.unsqueeze(0)).squeeze(0)

#Getting the keypoints using Harris detector.
def get_keypoints(img, num_points = 500):
    gray = to_gray(img)
    harris = K.feature.harris_response(gray.unsqueeze(0), k = 0.04)
    harris = harris.squeeze()

    #Flattening and picking the top responses
    vals, idx = torch.topk(harris.view(-1), num_points)
    ys = idx // harris.shape[1]
    xs = idx % harris.shape[1]

    return torch.stack([xs, ys], dim = 1).float()

#Extracting normalized local patch descriptors around each keypoint.
def extract_pathces(img, kps, patch_size=5):
    pad = patch_size // 2
    img_pad = torch.nn.functional.pad(
        img.unsqueeze(0), (pad, pad, pad, pad), mode='reflect'
    ).squeeze(0)

    patches = []
    valid_kps = []

    for kp in kps:
        x, y = int(kp[0].item()), int(kp[1].item())

        x_pad = x + pad
        y_pad = y + pad

        patch = img_pad[:, y_pad-pad:y_pad+pad+1, x_pad-pad:x_pad+pad+1]

        if patch.shape[1] != patch_size or patch.shape[2] != patch_size:
            continue

        patch = patch.flatten().float()
        patch = patch - patch.mean()
        norm = torch.norm(patch)
        if norm > 1e-8:
            patch = patch / norm

        patches.append(patch)
        valid_kps.append(kp)

    if len(patches) == 0:
        return torch.empty((0, img.shape[0] * patch_size * patch_size)), torch.empty((0, 2))

    return torch.stack(patches), torch.stack(valid_kps).float()

#Matching feature descriptors between two images using nearest-neighbor distance.
def match_features(desc1, desc2, threshold=0.9):
    matches = []

    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
        return matches

    dmat = torch.cdist(desc1, desc2)

    nn12_vals, nn12_idx = torch.min(dmat, dim=1)
    nn21_vals, nn21_idx = torch.min(dmat, dim=0)

    for i in range(desc1.shape[0]):
        j = nn12_idx[i].item()
        if nn21_idx[j].item() == i and nn12_vals[i].item() < threshold:
            matches.append((i, j))

    return matches

#Computing homography matrix using Direct Linear Transformation (DLT) from point correspondences.
def compute_homography(src, dst):
    if src.shape[0] < 4:
        return None

    A = []

    for i in range(src.shape[0]):
        x, y = src[i, 0], src[i, 1]
        xp, yp = dst[i, 0], dst[i, 1]

        A.append(torch.tensor(
            [-x, -y, -1.0, 0.0, 0.0, 0.0, x * xp, y * xp, xp],
            dtype=torch.float32
        ))
        A.append(torch.tensor(
            [0.0, 0.0, 0.0, -x, -y, -1.0, x * yp, y * yp, yp],
            dtype=torch.float32
        ))

    A = torch.stack(A, dim=0)

    try:
        _, _, Vh = torch.linalg.svd(A)
        H = Vh[-1].reshape(3, 3)
        if torch.abs(H[-1, -1]) < 1e-8:
            return None
        H = H / H[-1, -1]
        return H
    except:
        return None

# Estimating homography using RANSAC by selecting the model with maximum inliers
def ransac_homography(src_pts, dst_pts, iterations=1500, thresh=3.0):
    best_H = None
    best_inliers = None
    best_inlier_count = 0

    if len(src_pts) < 4:
        return None

    for _ in range(iterations):
        idx = torch.randperm(len(src_pts))[:4]
        H = compute_homography(src_pts[idx], dst_pts[idx])

        if H is None or torch.isnan(H).any() or torch.isinf(H).any():
            continue

        src_h = torch.cat([src_pts, torch.ones(len(src_pts), 1)], dim=1)
        proj = (H @ src_h.T).T

        denom = proj[:, 2:]
        valid = torch.abs(denom.squeeze()) > 1e-6
        if valid.sum() < 4:
            continue

        proj_xy = torch.zeros_like(dst_pts)
        proj_xy[valid] = proj[valid, :2] / denom[valid]

        errors = torch.full((len(src_pts),), 1e9)
        errors[valid] = torch.norm(proj_xy[valid] - dst_pts[valid], dim=1)

        inliers = errors < thresh
        inlier_count = int(inliers.sum().item())

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_H = H
            best_inliers = inliers

    if best_H is None or best_inliers is None:
        return None

    if int(best_inliers.sum().item()) >= 4:
        refined_H = compute_homography(src_pts[best_inliers], dst_pts[best_inliers])
        if refined_H is not None and not torch.isnan(refined_H).any() and not torch.isinf(refined_H).any():
            best_H = refined_H

    return best_H

def to_uint8_image(img):
    if img.max() <= 1.0:
        return (img.clamp(0, 1) * 255.0).to(torch.uint8)
    return img.clamp(0, 255).to(torch.uint8)

# Applying homography transformation to a set of 2D points
def apply_homography(H, pts):
    ones = torch.ones((pts.shape[0], 1), dtype=pts.dtype, device=pts.device)
    pts_h = torch.cat([pts, ones], dim=1)
    warped = (H @ pts_h.T).T
    denom = warped[:, 2:]
    denom = torch.where(torch.abs(denom) < 1e-6, torch.ones_like(denom), denom)
    return warped[:, :2] / denom

# Warping image into a new coordinate frame using homography
def warp_image(img, H, out_shape):
    B, C, Ht, Wt = 1, img.shape[0], out_shape[0], out_shape[1]
    grid = K.geometry.transform.warp_perspective(
        img.unsqueeze(0),
        H.unsqueeze(0),
        dsize = (Ht, Wt)
    )
    return grid.squeeze(0)

# Checking if a homography is valid based on scale, skew, and numerical stability.
def is_reasonable_homography(H):
    if H is None:
        return False
    if torch.isnan(H).any() or torch.isinf(H).any():
        return False

    if torch.abs(H[2, 2]) > 1e-8:
        H = H / H[2, 2]

    A = H[:2, :2]
    det = torch.det(A).abs().item()

    # tighter than before
    if det < 0.2 or det > 5.0:
        return False

    # allow perspective but not extreme
    if torch.abs(H[2, 0]).item() > 0.01 or torch.abs(H[2, 1]).item() > 0.01:
        return False

    return True

# Generating center-weighted mask for smooth blending in panorama
def make_feather_weight(h, w):
    ys = torch.arange(h, dtype=torch.float32)
    xs = torch.arange(w, dtype=torch.float32)

    y_dist = torch.minimum(ys, torch.tensor(float(h - 1)) - ys)
    x_dist = torch.minimum(xs, torch.tensor(float(w - 1)) - xs)

    wy = (y_dist + 1.0) / (y_dist.max() + 1.0 + 1e-6)
    wx = (x_dist + 1.0) / (x_dist.max() + 1.0 + 1e-6)

    weight = wy.unsqueeze(1) * wx.unsqueeze(0)
    return weight.unsqueeze(0)   # 1 x h x w

# RANSAC variant for Task 2 with stricter filtering for stable panorama stitching.
def ransac_homography_task2(src_pts, dst_pts, iterations=1500, thresh=3.0):
    best_H = None
    best_inliers = None
    best_count = 0

    if len(src_pts) < 4:
        return None

    for _ in range(iterations):
        idx = torch.randperm(len(src_pts))[:4]
        H = compute_homography(src_pts[idx], dst_pts[idx])

        if H is None:
            continue
        if torch.isnan(H).any() or torch.isinf(H).any():
            continue

        src_h = torch.cat([src_pts, torch.ones(len(src_pts), 1)], dim=1)
        proj = (H @ src_h.T).T

        denom = proj[:, 2:]
        valid = torch.abs(denom.squeeze()) > 1e-6
        if valid.sum() < 4:
            continue

        proj_xy = torch.zeros_like(dst_pts)
        proj_xy[valid] = proj[valid, :2] / denom[valid]

        errors = torch.full((len(src_pts),), 1e9)
        errors[valid] = torch.norm(proj_xy[valid] - dst_pts[valid], dim=1)

        inliers = errors < thresh
        count = int(inliers.sum().item())

        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_H = H

    if best_H is None or best_inliers is None:
        return None

    if int(best_inliers.sum().item()) >= 4:
        refined_H = compute_homography(src_pts[best_inliers], dst_pts[best_inliers])
        if refined_H is not None and not torch.isnan(refined_H).any() and not torch.isinf(refined_H).any():
            return refined_H

    return best_H

# Blending two images by selecting consistent regions.
def blen(img1, img2):
    mask1 = (img1.sum(0) > 0)
    
    result = img1.clone()
    
    # wherever img1 is empty → take img2
    result[:, ~mask1] = img2[:, ~mask1]
    
    return result

# ------------------------------------ Task 1 ------------------------------------ #
def stitch_background(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: input images are a dict of 2 images of torch.Tensor represent an input images for task-1.
    Returns:
        img: stitched_image: torch.Tensor of the output image.
    """
    img = torch.zeros((3, 256, 256)) # assumed 256*256 resolution. Update this as per your logic.

    #TODO: Add your code here. Do not modify the return and input arguments.

    # Initializing keys and loading the images and converting them to float for processing
    keys = list(imgs.keys())
    img1 = imgs[keys[0]].float()
    img2 = imgs[keys[1]].float()

    # Detecting the Harris corner keypoints in both the images
    kp1 = get_keypoints(img1, num_points=800)
    kp2 = get_keypoints(img2, num_points=800)

    # Extracting normalized local patch descriptors around the detected keypoints
    desc1, kp1 = extract_pathces(img1, kp1)
    desc2, kp2 = extract_pathces(img2, kp2)

    # Matching the descriptors between the two images using nearest-neighbor similarity
    matches = match_features(desc1, desc2, threshold=1.4)

    if len(matches) < 4:
        return img1.clamp(0, 255).to(torch.uint8)

    # Estimating a robust homography between the two images using RANSAC.
    src_pts = torch.stack([kp1[i] for i, _ in matches], dim = 0)
    dst_pts = torch.stack([kp2[j] for _, j in matches], dim = 0)

    # Inverting the homography so the second image can be warped into the first image's frame.
    H = ransac_homography(src_pts, dst_pts)
    if H is None or torch.isnan(H).any() or torch.isinf(H).any():
        return img1.clamp(0, 255).to(torch.uint8)
    try:
        H_inv = torch.inverse(H)
    except:
        return img1.clamp(0, 255).to(torch.uint8)
    if torch.isnan(H_inv).any() or torch.isinf(H_inv).any():
        return img1.clamp(0, 255).to(torch.uint8)

    _, h1, w1 = img1.shape
    _, h2, w2 = img2.shape

    # Defining the four corners of the first image.
    corners1 = torch.tensor([
        [0.0, 0.0],
        [w1 - 1.0, 0.0],
        [0.0, h1 - 1.0],
        [w1 - 1.0, h1 - 1.0] 
    ], dtype=torch.float32)

    # Defining corners of the second image
    corners2 = torch.tensor([
        [0.0, 0.0],
        [w2 - 1.0, 0.0],
        [0.0, h2 - 1.0],
        [w2 - 1.0, h2 - 1.0] 
    ], dtype=torch.float32)

    # Converting the second-image's corners to homogeneous form and warp them using the inverse homography.
    ones2 = torch.ones((corners2.shape[0], 1), dtype=torch.float32)
    corners2_h = torch.cat([corners2, ones2], dim=1)
    warped_corners2 = (H_inv @ corners2_h.T).T
    denom = warped_corners2[:, 2:]
    denom = torch.where(torch.abs(denom) < 1e-6, torch.ones_like(denom), denom)
    warped_corners2 = warped_corners2[:, :2] / denom

    all_corners = torch.cat([corners1, warped_corners2], dim=0)

    # Minimum and maximum x/y values needed for the output canvas
    min_xy = torch.floor(all_corners.min(dim=0).values)
    max_xy = torch.ceil(all_corners.max(dim=0).values)

    min_x, min_y = min_xy[0].item(), min_xy[1].item()
    max_x, max_y = max_xy[0].item(), max_xy[1].item()

    # Computing the final panorama width and height.
    out_w = int(max_x - min_x + 1)
    out_h = int(max_y - min_y + 1)

    # Rejecting invalid or unreasonably large ouput canvas sizes.
    if out_w <= 0 or out_h <= 0 or out_w > 4000 or out_h > 4000:
        return img1.clamp(0, 255).to(torch.uint8)

    #Translation offsets
    tx = -min_x
    ty = -min_y

    # Translation matrix to shift the stitched result into the output canvas.
    T = torch.tensor([
        [1.0, 0.0, tx],
        [0.0, 1.0, ty],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)

    warp1 = warp_image(img1, T, (out_h, out_w))
    warp2 = warp_image(img2, T @ H_inv, (out_h, out_w))

    img = blen(warp1, warp2)

    # Findfing the valid stitched region and cropping the empty black borders.
    mask = (img.sum(dim=0) > 0)
    ys, xs = torch.where(mask)

    if len(xs) > 0 and len(ys) > 0:
        x_min, x_max = xs.min().item(), xs.max().item()
        y_min, y_max = ys.min().item(), ys.max().item()
        img = img[:, y_min:y_max+1, x_min:x_max+1]

    # Converting the final image to uint8
    img = img.clamp(0, 255).to(torch.uint8)
    return img

# ------------------------------------ Task 2 ------------------------------------ #
def panorama(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: dict {filename: CxHxW tensor} for task-2.
    Returns:
        img: panorama, 
        overlap: torch.Tensor of the output image. 
    """
    img = torch.zeros((3, 256, 256)) # assumed 256*256 resolution. Update this as per your logic.

    #TODO: Add your code here. Do not modify the return and input arguments.

    # Storing image names in a list and counting the number of images
    keys = list(imgs.keys())
    n = len(keys)

    # Defining a local feature-matching helper that uses a Lowe-style ratio test and mutual nearest-neighbor consistency.
    def match_features_task2(desc1, desc2, ratio=0.75):
        matches = []
        if desc1.shape[0] < 2 or desc2.shape[0] < 2:
            return matches

        dmat = torch.cdist(desc1, desc2)

        vals12, idx12 = torch.topk(dmat, k=2, largest=False, dim=1)
        _, idx21 = torch.min(dmat, dim=0)

        for i in range(desc1.shape[0]):
            d1 = vals12[i, 0].item()
            d2 = vals12[i, 1].item()
            j = idx12[i, 0].item()

            # Lowe ratio + mutual consistency
            if d1 < ratio * d2 and idx21[j].item() == i:
                matches.append((i, j))

        return matches

    # Defining a helper to count the number of points agreeing with a homography
    def count_inliers(H, src_pts, dst_pts, thresh=2.5):
        if H is None:
            return 0, None

        src_h = torch.cat([src_pts, torch.ones(len(src_pts), 1)], dim=1)
        proj = (H @ src_h.T).T

        denom = proj[:, 2:]
        valid = torch.abs(denom.squeeze()) > 1e-6
        if valid.sum() < 4:
            return 0, None

        proj_xy = torch.zeros_like(dst_pts)
        proj_xy[valid] = proj[valid, :2] / denom[valid]

        errors = torch.full((len(src_pts),), 1e9)
        errors[valid] = torch.norm(proj_xy[valid] - dst_pts[valid], dim=1)

        inliers = errors < thresh
        return int(inliers.sum().item()), inliers

    # Extracting keyppints and patch descriptors from every image in the input set
    features = []
    for i in range(n):
        img = imgs[keys[i]].float()
        kp = get_keypoints(img, num_points=1200)
        desc, kp = extract_pathces(img, kp)
        features.append((desc, kp))
    # Overlap matrix and a dictionary to store pairwise homographies
    overlap = torch.zeros((n, n), dtype=torch.float32)
    homographies = {}

    # Comparing every image pair, matching features, estimating homography, and recording valid overlaps
    for i in range(n):
        desc1, kp1 = features[i]

        for j in range(i + 1, n):
            desc2, kp2 = features[j]

            matches = match_features_task2(desc1, desc2, ratio=0.8)
            if len(matches) < 12:
                continue

            src_pts = torch.stack([kp1[a] for a, _ in matches], dim=0)
            dst_pts = torch.stack([kp2[b] for _, b in matches], dim=0)

            H_ij = ransac_homography_task2(src_pts, dst_pts, iterations=2000, thresh=2.5)
            if H_ij is None or torch.isnan(H_ij).any() or torch.isinf(H_ij).any():
                continue

            inlier_count, inliers = count_inliers(H_ij, src_pts, dst_pts, thresh=2.5)
            if inlier_count < 10:
                continue

            # Refining Inliers
            H_ij = compute_homography(src_pts[inliers], dst_pts[inliers])
            if H_ij is None or torch.isnan(H_ij).any() or torch.isinf(H_ij).any():
                continue

            if not is_reasonable_homography(H_ij):
                continue
            try:
                H_ji = torch.inverse(H_ij)
            except:
                continue
            if torch.isnan(H_ji).any() or torch.isinf(H_ji).any():
                continue

            if torch.abs(H_ij[2, 2]) > 1e-8:
                H_ij = H_ij / H_ij[2, 2]
            if torch.abs(H_ji[2, 2]) > 1e-8:
                H_ji = H_ji / H_ji[2, 2]

            homographies[(i, j)] = H_ij
            homographies[(j, i)] = H_ji

            overlap[i, j] = float(inlier_count)
            overlap[j, i] = float(inlier_count)

    # Choosing REference image as the one with strongest total overlap with other images
    row_scores = overlap.sum(dim=1)
    ref = int(torch.argmax(row_scores).item())

    global_H = {ref: torch.eye(3, dtype=torch.float32)}
    used = set([ref])

    while True:
        best_edge = None
        best_score = -1.0

        for u in list(used):
            for v in range(n):
                if v in used or u == v:
                    continue
                if (v, u) in homographies:
                    score = overlap[v, u].item()
                    if score > best_score:
                        best_score = score
                        best_edge = (u, v)

        # No more reliable connection condition
        if best_edge is None or best_score < 5:
            break

        u, v = best_edge

        H_v_to_u = homographies[(v, u)]
        H_v_to_ref = global_H[u] @ H_v_to_u

        if torch.isnan(H_v_to_ref).any() or torch.isinf(H_v_to_ref).any():
            overlap[v, u] = 0
            overlap[u, v] = 0
            continue

        if torch.abs(H_v_to_ref[2, 2]) > 1e-8:
            H_v_to_ref = H_v_to_ref / H_v_to_ref[2, 2]

        global_H[v] = H_v_to_ref
        used.add(v)

    if len(global_H) == 0:
        img = imgs[keys[ref]].float()
        return img.clamp(0, 255).to(torch.uint8), overlap

    all_corners = []
    for i in global_H:
        img_i = imgs[keys[i]].float()
        _, h, w = img_i.shape

        corners = torch.tensor([
            [0.0, 0.0],
            [w - 1.0, 0.0],
            [0.0, h - 1.0],
            [w - 1.0, h - 1.0]
        ], dtype=torch.float32)

        warped_corners = apply_homography(global_H[i], corners)
        all_corners.append(warped_corners)

    all_corners = torch.cat(all_corners, dim=0)

    min_xy = torch.floor(all_corners.min(dim=0).values)
    max_xy = torch.ceil(all_corners.max(dim=0).values)

    min_x, min_y = min_xy[0].item(), min_xy[1].item()
    max_x, max_y = max_xy[0].item(), max_xy[1].item()

    out_w = int(max_x - min_x + 1)
    out_h = int(max_y - min_y + 1)

    if out_w <= 0 or out_h <= 0 or out_w > 5000 or out_h > 5000:
        img = imgs[keys[ref]].float()
        return img.clamp(0, 255).to(torch.uint8), overlap

    T = torch.tensor([
        [1.0, 0.0, -min_x],
        [0.0, 1.0, -min_y],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)

    # Warping + blending
    canvas_sum = torch.zeros((3, out_h, out_w), dtype=torch.float32)
    canvas_weight = torch.zeros((1, out_h, out_w), dtype=torch.float32)

    ordered_ids = [ref] + sorted([i for i in global_H if i != ref])

    for i in ordered_ids:
        img_i = imgs[keys[i]].float()
        _, h_i, w_i = img_i.shape

        H_i = T @ global_H[i]

        if torch.abs(H_i[2, 2]) > 1e-8:
            H_i = H_i / H_i[2, 2]

        if torch.isnan(H_i).any() or torch.isinf(H_i).any():
            continue

        warped = warp_image(img_i, H_i, (out_h, out_w))

        mask = (warped.sum(dim=0, keepdim=True) > 0).float()

        local_weight = make_feather_weight(h_i, w_i)
        warped_weight = warp_image(local_weight, H_i, (out_h, out_w))
        
        warped_weight = warped_weight * mask

        canvas_sum += warped * warped_weight
        canvas_weight += warped_weight

    zero_mask = canvas_weight <= 1e-6
    canvas_weight = torch.clamp(canvas_weight, min=1e-6)

    img = canvas_sum / canvas_weight
    img = img * (~zero_mask).float()

    mask = (img.sum(dim=0) > 0)
    ys, xs = torch.where(mask)

    if len(xs) > 0 and len(ys) > 0:
        x_min, x_max = xs.min().item(), xs.max().item()
        y_min, y_max = ys.min().item(), ys.max().item()
        img = img[:, y_min:y_max+1, x_min:x_max+1]

    img = img.clamp(0, 255).to(torch.uint8)
    # Converting overlapping scored to a binary overlap matrix as instructed in the requirements. NxN One-hot overlap array
    overlap_bin = (overlap > 0).float()
    return img, overlap_bin