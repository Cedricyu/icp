import numpy as np
import pycolmap

def apply_distortion(extra_params, u, v):
    """
    Applies radial or OpenCV distortion to the given 2D points.

    Args:
        extra_params (torch.Tensor or numpy.ndarray): Distortion parameters of shape BxN, where N can be 1, 2, or 4.
        u (torch.Tensor or numpy.ndarray): Normalized x coordinates of shape Bxnum_tracks.
        v (torch.Tensor or numpy.ndarray): Normalized y coordinates of shape Bxnum_tracks.

    Returns:
        points2D (torch.Tensor): Distorted 2D points of shape BxNx2.
    """
    extra_params = _ensure_torch(extra_params)
    u = _ensure_torch(u)
    v = _ensure_torch(v)

    num_params = extra_params.shape[1]

    if num_params == 1:
        # Simple radial distortion
        k = extra_params[:, 0]
        u2 = u * u
        v2 = v * v
        r2 = u2 + v2
        radial = k[:, None] * r2
        du = u * radial
        dv = v * radial

    elif num_params == 2:
        # RadialCameraModel distortion
        k1, k2 = extra_params[:, 0], extra_params[:, 1]
        u2 = u * u
        v2 = v * v
        r2 = u2 + v2
        radial = k1[:, None] * r2 + k2[:, None] * r2 * r2
        du = u * radial
        dv = v * radial

    elif num_params == 4:
        # OpenCVCameraModel distortion
        k1, k2, p1, p2 = (extra_params[:, 0], extra_params[:, 1], extra_params[:, 2], extra_params[:, 3])
        u2 = u * u
        v2 = v * v
        uv = u * v
        r2 = u2 + v2
        radial = k1[:, None] * r2 + k2[:, None] * r2 * r2
        du = u * radial + 2 * p1[:, None] * uv + p2[:, None] * (r2 + 2 * u2)
        dv = v * radial + 2 * p2[:, None] * uv + p1[:, None] * (r2 + 2 * v2)
    else:
        raise ValueError("Unsupported number of distortion parameters")

    u = u.clone() + du
    v = v.clone() + dv

    return u, v

def img_from_cam_np(
    intrinsics: np.ndarray, points_cam: np.ndarray, extra_params: np.ndarray | None = None, default: float = 0.0
) -> np.ndarray:
    """
    Apply intrinsics (and optional radial distortion) to camera-space points.

    Args
    ----
    intrinsics  : (B,3,3) camera matrix K.
    points_cam  : (B,3,N) homogeneous camera coords  (x, y, z)ᵀ.
    extra_params: (B, N) or (B, k) distortion params (k = 1,2,4) or None.
    default     : value used for np.nan replacement.

    Returns
    -------
    points2D : (B,N,2) pixel coordinates.
    """
    # 1. perspective divide  ───────────────────────────────────────
    z = points_cam[:, 2:3, :]  # (B,1,N)
    points_cam_norm = points_cam / z  # (B,3,N)
    uv = points_cam_norm[:, :2, :]  # (B,2,N)

    # 2. optional distortion ──────────────────────────────────────
    if extra_params is not None:
        uu, vv = apply_distortion(extra_params, uv[:, 0], uv[:, 1])
        uv = np.stack([uu, vv], axis=1)  # (B,2,N)

    # 3. homogeneous coords then K multiplication ─────────────────
    ones = np.ones_like(uv[:, :1, :])  # (B,1,N)
    points_cam_h = np.concatenate([uv, ones], axis=1)  # (B,3,N)

    # batched mat-mul: K · [u v 1]ᵀ
    points2D_h = np.einsum("bij,bjk->bik", intrinsics, points_cam_h)  # (B,3,N)
    points2D = np.nan_to_num(points2D_h[:, :2, :], nan=default)  # (B,2,N)

    return points2D.transpose(0, 2, 1)  # (B,N,2)


def project_3D_points_np(
    points3D: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray | None = None,
    extra_params: np.ndarray | None = None,
    *,
    default: float = 0.0,
    only_points_cam: bool = False,
):
    """
    NumPy clone of ``project_3D_points``.

    Parameters
    ----------
    points3D          : (N,3) world-space points.
    extrinsics        : (B,3,4)  [R|t] matrix for each of B cameras.
    intrinsics        : (B,3,3)  K matrix (optional if you only need cam-space).
    extra_params      : (B,k) or (B,N) distortion parameters (k ∈ {1,2,4}) or None.
    default           : value used to replace NaNs.
    only_points_cam   : if True, skip the projection and return points_cam with points2D as None.

    Returns
    -------
    (points2D, points_cam) : A tuple where points2D is (B,N,2) pixel coords or None if only_points_cam=True,
                           and points_cam is (B,3,N) camera-space coordinates.
    """
    # ----- 0. prep sizes -----------------------------------------------------
    N = points3D.shape[0]  # #points
    B = extrinsics.shape[0]  # #cameras

    # ----- 1. world → homogeneous -------------------------------------------
    w_h = np.ones((N, 1), dtype=points3D.dtype)
    points3D_h = np.concatenate([points3D, w_h], axis=1)  # (N,4)

    # broadcast to every camera (no actual copying with np.broadcast_to) ------
    points3D_h_B = np.broadcast_to(points3D_h, (B, N, 4))  # (B,N,4)

    # ----- 2. apply extrinsics  (camera frame) ------------------------------
    # X_cam = E · X_hom
    # einsum:  E_(b i j)  ·  X_(b n j)  →  (b n i)
    points_cam = np.einsum("bij,bnj->bni", extrinsics, points3D_h_B)  # (B,N,3)
    points_cam = points_cam.transpose(0, 2, 1)  # (B,3,N)

    if only_points_cam:
        return None, points_cam

    # ----- 3. intrinsics + distortion ---------------------------------------
    if intrinsics is None:
        raise ValueError("`intrinsics` must be provided unless only_points_cam=True")

    points2D = img_from_cam_np(intrinsics, points_cam, extra_params=extra_params, default=default)

    return points2D, points_cam


def _build_pycolmap_intri(fidx, intrinsics, camera_type, extra_params=None):
    """
    Helper function to get camera parameters based on camera type.

    Args:
        fidx: Frame index
        intrinsics: Camera intrinsic parameters
        camera_type: Type of camera model
        extra_params: Additional parameters for certain camera types

    Returns:
        pycolmap_intri: NumPy array of camera parameters
    """
    if camera_type == "PINHOLE":
        pycolmap_intri = np.array(
            [intrinsics[fidx][0, 0], intrinsics[fidx][1, 1], intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]]
        )
    elif camera_type == "SIMPLE_PINHOLE":
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        pycolmap_intri = np.array([focal, intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]])
    elif camera_type == "SIMPLE_RADIAL":
        raise NotImplementedError("SIMPLE_RADIAL is not supported yet")
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        pycolmap_intri = np.array([focal, intrinsics[fidx][0, 2], intrinsics[fidx][1, 2], extra_params[fidx][0]])
    else:
        raise ValueError(f"Camera type {camera_type} is not supported yet")

    return pycolmap_intri


# %%
def batch_np_matrix_to_pycolmap(
    points3d,
    extrinsics,
    intrinsics,
    tracks,
    image_size,
    masks=None,
    max_reproj_error=None,
    max_points3D_val=3000,
    shared_camera=False,
    camera_type="SIMPLE_PINHOLE",
    extra_params=None,
    min_inlier_per_frame=30,
    image_list=None,
    points_rgb=None,
):
    """
    Convert Batched NumPy Arrays to PyCOLMAP

    Check https://github.com/colmap/pycolmap for more details about its format

    NOTE that colmap expects images/cameras/points3D to be 1-indexed
    so there is a +1 offset between colmap index and batch index


    NOTE: different from VGGSfM, this function:
    1. Use np instead of torch
    2. Frame index and camera id starts from 1 rather than 0 (to fit the format of PyCOLMAP)
    """
    # points3d: Px3
    # extrinsics: Nx3x4
    # intrinsics: Nx3x3
    # tracks: NxPx2
    # masks: NxP
    # image_size: 2, assume all the frames have been padded to the same size
    # where N is the number of frames and P is the number of tracks

    N, P, _ = tracks.shape
    assert len(extrinsics) == N
    assert len(intrinsics) == N
    assert len(points3d) == P
    assert image_size.shape[0] == 2

    reproj_mask = None

    if max_reproj_error is not None:
        projected_points_2d, projected_points_cam = project_3D_points_np(points3d, extrinsics, intrinsics)
        projected_diff = np.linalg.norm(projected_points_2d - tracks, axis=-1)
        projected_points_2d[projected_points_cam[:, -1] <= 0] = 1e6
        reproj_mask = projected_diff < max_reproj_error

    if masks is not None and reproj_mask is not None:
        masks = np.logical_and(masks, reproj_mask)
    elif masks is not None:
        masks = masks
    else:
        masks = reproj_mask

    assert masks is not None

    inliers_per_frame = masks.sum(1)
    print(f"[DEBUG] Final inliers per frame: {inliers_per_frame}")
    print(f"[DEBUG] Min inliers per frame: {inliers_per_frame.min()}, Max: {inliers_per_frame.max()}")

    if inliers_per_frame.min() < min_inlier_per_frame:
        print(f"[WARNING] Some frames have < {min_inlier_per_frame} inliers, skip BA.")
        return None, None


    # Reconstruction object, following the format of PyCOLMAP/COLMAP
    reconstruction = pycolmap.Reconstruction()

    inlier_num = masks.sum(0)
    valid_mask = inlier_num >= 2  # a track is invalid if without two inliers
    valid_idx = np.nonzero(valid_mask)[0]

    # Only add 3D points that have sufficient 2D points
    for vidx in valid_idx:
        pt = np.asarray(points3d[vidx], dtype=np.float64).reshape(3)
        rgb = np.asarray(points_rgb[vidx], dtype=np.uint8).reshape(3)
        pt = np.ascontiguousarray(pt)
        rgb = np.ascontiguousarray(rgb)
        track = pycolmap.Track()
        reconstruction.add_point3D(pt, track, rgb)


    num_points3D = len(valid_idx)
    camera = None
    # frame idx
    for fidx in range(N):
        # set camera
        if camera is None or (not shared_camera):
            pycolmap_intri = _build_pycolmap_intri(fidx, intrinsics, camera_type, extra_params)

            camera = pycolmap.Camera(
                model=camera_type, width=image_size[0], height=image_size[1], params=pycolmap_intri, camera_id=fidx + 1
            )

            # add camera
            reconstruction.add_camera(camera)

        # set image
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsics[fidx][:3, :3]), extrinsics[fidx][:3, 3]
        )  # Rot and Trans

        image = pycolmap.Image(
            id=fidx + 1, name=image_list[fidx], camera_id=camera.camera_id, cam_from_world=cam_from_world
        )

        points2D_list = []

        point2D_idx = 0

        # NOTE point3D_id start by 1
        for point3D_id in range(1, num_points3D + 1):
            original_track_idx = valid_idx[point3D_id - 1]

            if (reconstruction.points3D[point3D_id].xyz < max_points3D_val).all():
                if masks[fidx][original_track_idx]:
                    # It seems we don't need +0.5 for BA
                    point2D_xy = tracks[fidx][original_track_idx]
                    # Please note when adding the Point2D object
                    # It not only requires the 2D xy location, but also the id to 3D point
                    points2D_list.append(pycolmap.Point2D(point2D_xy, point3D_id))

                    # add element
                    track = reconstruction.points3D[point3D_id].track
                    track.add_element(fidx + 1, point2D_idx)
                    point2D_idx += 1

        assert point2D_idx == len(points2D_list)

        try:
            image.points2D = pycolmap.ListPoint2D(points2D_list)
            image.registered = True
        except:
            print(f"frame {fidx + 1} is out of BA")
            image.registered = False

        # add image
        reconstruction.add_image(image)

    return reconstruction, valid_mask
