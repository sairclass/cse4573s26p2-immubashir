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

    keys = list(imgs.keys())
    img1 = imgs[keys[0]].float()
    img2 = imgs[keys[1]].float()

    kp1 = get_keypoints(img1, num_points=800)
    kp2 = get_keypoints(img2, num_points=800)

    desc1, kp1 = extract_pathces(img1, kp1)
    desc2, kp2 = extract_pathces(img2, kp2)

    matches = match_features(desc1, desc2, threshold=1.4)

    # clamping the pixel values to be in the range [0, 255] and converting to uint8 as it is required as the output format.
    if len(matches) < 4:
        return img1.clamp(0, 255).to(torch.uint8)
    
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
    overlap = torch.empty((3, 256, 256)) # assumed empty 256*256 overlap. Update this as per your logic.

    #TODO: Add your code here. Do not modify the return and input arguments.

    return img, overlap