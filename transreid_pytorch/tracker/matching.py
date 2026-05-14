import numpy as np
import scipy.linalg
from scipy.optimize import linear_sum_assignment

def iou_batch(bboxes1, bboxes2):
    """
    Computes IOU between two sets of bounding boxes.
    bboxes: [[x1, y1, x2, y2], ...]
    Returns an NxM matrix of IoU values.
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h

    o = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1]) +
              (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)
    return o

def iou_distance(atracks, btracks):
    """
    Computes 1 - IOU distance between two lists of tracks (or detections).
    Input objects must have a `.tlbr` property returning [x1, y1, x2, y2].
    """
    if (len(atracks) == 0) or (len(btracks) == 0):
        return np.zeros((len(atracks), len(btracks)))
    bboxes1 = np.array([track.tlbr for track in atracks], dtype=np.float32)
    bboxes2 = np.array([track.tlbr for track in btracks], dtype=np.float32)
    ious = iou_batch(bboxes1, bboxes2)
    return 1 - ious

def embedding_distance(tracks, detections, metric='cosine'):
    """
    Computes cosine distance between tracks and detections embeddings.
    """
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)))
    
    track_features = np.array([track.smooth_feat for track in tracks])
    det_features = np.array([det.curr_feat for det in detections])
    
    # Cosine distance: 1 - sum(A * B) (assuming vectors are already L2 normalized)
    cost_matrix = 1.0 - np.dot(track_features, det_features.T)
    cost_matrix = np.maximum(0.0, cost_matrix)
    return cost_matrix

def linear_assignment(cost_matrix, thresh):
    """
    Linear assignment wrapper.
    Returns:
        matches: list of (row_idx, col_idx)
        unmatched_a: list of unmatched row indices
        unmatched_b: list of unmatched column indices
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = np.zeros(cost_matrix.shape), np.zeros(cost_matrix.shape[0]), np.zeros(cost_matrix.shape[1])
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > thresh:
            unmatched_a.append(r)
            unmatched_b.append(c)
        else:
            matches.append((r, c))
            
    unmatched_a += list(set(range(cost_matrix.shape[0])) - set(row_ind))
    unmatched_b += list(set(range(cost_matrix.shape[1])) - set(col_ind))
    
    return np.array(matches, dtype=int).reshape(-1, 2), np.array(unmatched_a, dtype=int), np.array(unmatched_b, dtype=int)
