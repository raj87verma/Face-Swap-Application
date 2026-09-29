"""
Classical landmark-based face swap engine.

Pipeline (no deep-learning / generative model involved):
  1. Take source face landmarks (from a reference image) and destination
     face landmarks (from a video frame).
  2. Build a Delaunay triangulation over the destination landmarks.
  3. For each triangle, compute an affine warp mapping the corresponding
     source triangle onto the destination triangle (classical geometric
     image warping).
  4. Assemble the warped source face and blend it onto the destination
     frame using a convex-hull mask + Poisson/seamless cloning
     (cv2.seamlessClone) for natural lighting/skin-tone blending.

This is the same family of technique used by classic "face swap" scripts
that predate deep generative models (e.g. the well-known
matthewearl/faceswap and learnopencv Delaunay-triangulation approaches).
"""
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.utils.config import FACE_SWAP_POINTS


def _get_triangle_indices(points: np.ndarray, rect: Tuple[int, int, int, int]) -> List[Tuple[int, int, int]]:
    """Delaunay-triangulate `points` (bounded by `rect`) and return, for each
    triangle, the indices (into `points`) of its three vertices."""
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert((float(p[0]), float(p[1])))
    triangle_list = subdiv.getTriangleList()

    # Map each triangle's 3 vertex coordinates back to point indices.
    point_index = {(int(round(p[0])), int(round(p[1]))): i for i, p in enumerate(points)}

    def _closest_index(pt):
        key = (int(round(pt[0])), int(round(pt[1])))
        if key in point_index:
            return point_index[key]
        # Fallback: nearest point (guards against float rounding mismatches)
        dists = np.sum((points - np.array(pt)) ** 2, axis=1)
        return int(np.argmin(dists))

    triangles = []
    for t in triangle_list:
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        x, y, w, h = rect
        if all(x <= p[0] <= x + w and y <= p[1] <= y + h for p in pts):
            idx = tuple(_closest_index(p) for p in pts)
            if len(set(idx)) == 3:
                triangles.append(idx)
    return triangles


def _warp_triangle(src_img: np.ndarray, dst_img: np.ndarray,
                    src_tri: np.ndarray, dst_tri: np.ndarray) -> None:
    """Affine-warp the triangular region `src_tri` from src_img into the
    triangular region `dst_tri` of dst_img (in place, alpha-blended by mask)."""
    src_rect = cv2.boundingRect(src_tri.astype(np.float32))
    dst_rect = cv2.boundingRect(dst_tri.astype(np.float32))

    sx, sy, sw, sh = src_rect
    dx, dy, dw, dh = dst_rect
    if sw <= 0 or sh <= 0 or dw <= 0 or dh <= 0:
        return

    src_tri_offset = src_tri - [sx, sy]
    dst_tri_offset = dst_tri - [dx, dy]

    src_crop = src_img[sy:sy + sh, sx:sx + sw]
    if src_crop.size == 0:
        return

    warp_mat = cv2.getAffineTransform(src_tri_offset.astype(np.float32),
                                       dst_tri_offset.astype(np.float32))
    warped = cv2.warpAffine(src_crop, warp_mat, (dw, dh), None,
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    mask = np.zeros((dh, dw), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(dst_tri_offset), 255)

    dst_region = dst_img[dy:dy + dh, dx:dx + dw]
    if dst_region.shape[:2] != warped.shape[:2]:
        return
    mask_3c = cv2.merge([mask, mask, mask]) if dst_img.ndim == 3 else mask
    np.copyto(dst_region, warped, where=mask_3c.astype(bool))


class FaceSwapper:
    """Warps a source face (from a reference image, with its own landmarks)
    onto a destination frame at the location/shape given by destination
    landmarks, then seamlessly blends it in."""

    def __init__(self, swap_point_indices: Optional[List[int]] = None,
                 feather_amount: int = 15):
        self.swap_point_indices = swap_point_indices or FACE_SWAP_POINTS
        self.feather_amount = feather_amount

    def swap(self, dest_frame: np.ndarray, dest_landmarks: np.ndarray,
              src_face_img: np.ndarray, src_landmarks: np.ndarray) -> np.ndarray:
        """Return a copy of dest_frame with the destination face region
        replaced by the (geometrically warped) source face."""
        dst_pts = dest_landmarks[self.swap_point_indices].astype(np.float64)
        src_pts = src_landmarks[self.swap_point_indices].astype(np.float64)

        h, w = dest_frame.shape[:2]
        rect = (0, 0, w, h)
        triangles = _get_triangle_indices(dst_pts, rect)
        if not triangles:
            return dest_frame

        warped_face = dest_frame.copy()
        for (i1, i2, i3) in triangles:
            src_tri = np.array([src_pts[i1], src_pts[i2], src_pts[i3]], dtype=np.float32)
            dst_tri = np.array([dst_pts[i1], dst_pts[i2], dst_pts[i3]], dtype=np.float32)
            _warp_triangle(src_face_img, warped_face, src_tri, dst_tri)

        # Build a feathered convex-hull mask over the destination face area
        # for seamless (Poisson) blending -- corrects for lighting/skin tone
        # differences between the reference image and the video frame.
        hull = cv2.convexHull(dst_pts.astype(np.int32))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)

        x, y, mw, mh = cv2.boundingRect(hull)
        center = (x + mw // 2, y + mh // 2)
        if mw <= 1 or mh <= 1:
            return dest_frame

        try:
            output = cv2.seamlessClone(warped_face, dest_frame, mask, center, cv2.NORMAL_CLONE)
        except cv2.error:
            # seamlessClone can fail on degenerate masks near frame edges;
            # fall back to a simple feathered alpha blend.
            blurred_mask = cv2.GaussianBlur(mask, (self.feather_amount | 1,) * 2, 0)
            alpha = (blurred_mask.astype(np.float32) / 255.0)[..., None]
            output = (warped_face.astype(np.float32) * alpha +
                      dest_frame.astype(np.float32) * (1 - alpha)).astype(np.uint8)
        return output

    def pick_best_reference(self, dest_landmarks: np.ndarray,
                             reference_landmarks_list: List[np.ndarray]) -> int:
        """Given the destination face's pose (approximated via landmark
        geometry), pick the index of the reference (from multiple available
        angle images) whose head pose most closely matches -- so a
        turned/tilted face in the video is matched with the closest angle
        reference rather than always the frontal one.

        Pose proxy: yaw approximated by relative distances of each eye/jaw
        edge to the nose tip; roll approximated by eye-line angle.
        """
        def pose_signature(lm: np.ndarray) -> np.ndarray:
            nose_tip = lm[30]
            left_jaw, right_jaw = lm[0], lm[16]
            left_eye_c = lm[42:48].mean(axis=0)
            right_eye_c = lm[36:42].mean(axis=0)

            d_left = np.linalg.norm(nose_tip - left_jaw)
            d_right = np.linalg.norm(nose_tip - right_jaw)
            yaw = (d_left - d_right) / (d_left + d_right + 1e-6)

            eye_vec = left_eye_c - right_eye_c
            roll = np.arctan2(eye_vec[1], eye_vec[0])
            return np.array([yaw, roll])

        dest_sig = pose_signature(dest_landmarks)
        best_idx, best_dist = 0, float("inf")
        for i, ref_lm in enumerate(reference_landmarks_list):
            ref_sig = pose_signature(ref_lm)
            dist = float(np.linalg.norm(dest_sig - ref_sig))
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx
