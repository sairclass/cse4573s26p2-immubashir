# Outputs

Your task-1, task-2, bonus_task outputs goes here.

Types: 

* PNG images of the stitched images.
* Text files of the overlap arrays.

Task-1:
File Type: PNG
Desc: Stitched background image generated from 2 input images using homography and combining them to produce a single image while minimizing the effect of moving foreground objects.

Task-2:
File Type: PNG
Desc: Output is a panorama created by stitching multiple overlapping images into a single wide-view image using feature mapping, RANSAC-based homography estimation, and blending.

Matrix Output:
Type: JSON / task2.json
Desc: Overlap matrix is an NxN binary matrix indicating which image pairs overlap. 
    1-> overlap
    0-> no overlap