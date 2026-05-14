# Offline Multi-Object Tracking via ReID and Spatio-Temporal Association

This directory contains the implementation of a two-phase, learning-free offline Multi-Object Tracker (MOT). The tracker is designed to leverage high-fidelity appearance embeddings (extracted via PersonViT) alongside a kinematic motion model (Kalman Filter) to resolve complex occlusion scenarios commonly found in crowded environments (e.g., DanceTrack). 

Because the tracking is performed *offline*, the algorithm can utilize future temporal information. Rather than relying entirely on online, causal frame-by-frame updates, the methodology follows a hierarchical association scheme:
1. **Local Association (Tracklet Generation):** Highly conservative online tracking to form pure, uncorrupted tracklets.
2. **Global Association (Tracklet Linking):** Bipartite matching over a tracklet-level graph to bridge long occlusion gaps.

---

## 1. System Architecture

The input to the system consists of a video sequence and a corresponding set of unassociated detection bounding boxes. 

### 1.1 Embedding Extraction
Each detection crop is passed through the `PersonViT` ReID model. The network outputs an $L_2$-normalized feature vector $\mathbf{e}_i \in \mathbb{R}^D$ for detection $i$. 

---

## 2. Phase 1: Local Association

The goal of the first phase is to generate highly reliable, short-term tracklets $\mathcal{T} = \{T_1, T_2, \dots, T_k\}$. A tracklet $T_k$ is a sequence of bounding boxes and ReID embeddings contiguous in time. 

To prevent identity switches (ID-switches), the local association is intentionally designed to be **conservative**. Whenever occlusion, crossing trajectories, or high ReID confusion occurs, the tracker terminates the tracklet rather than guessing the association. 

### 2.1 Motion Prediction
We employ an 8-dimensional Constant-Velocity Kalman Filter. The state space is defined as $\mathbf{x} = [c_x, c_y, a, h, v_x, v_y, v_a, v_h]^T$, where $(c_x, c_y)$ is the bounding box center, $a$ is the aspect ratio, and $h$ is the height. 
For each active tracklet $T_k$ at time $t$, the Kalman filter predicts the a priori state estimate $\mathbf{\hat{x}}_{k, t|t-1}$.

### 2.2 Cost Matrix Formulation
At frame $t$, let $\mathcal{D}_t$ be the set of new detections. We compute an association cost matrix $\mathbf{C} \in \mathbb{R}^{|\mathcal{T}_{active}| \times |\mathcal{D}_t|}$. The cost $C_{i, j}$ between tracklet $T_i$ and detection $D_j$ is a weighted combination of kinematic and appearance distances:

$$
C_{i, j} = \alpha \cdot d_{IoU}(T_i, D_j) + (1 - \alpha) \cdot d_{ReID}(T_i, D_j)
$$

Where:
- $d_{IoU}$ is the standard Intersection-over-Union distance ($1 - \text{IoU}$).
- $d_{ReID}$ is the cosine distance $1 - \langle \mathbf{\bar{e}}_i, \mathbf{e}_j \rangle$, where $\mathbf{\bar{e}}_i$ is the exponentially moving average (EMA) of the embeddings in tracklet $T_i$.

**Gating Mechanism:** To ensure purity, we apply a strict gating matrix $G_{i,j} \in \{0, \infty\}$:
$$
C_{i, j} = \begin{cases} 
C_{i,j} & \text{if } \text{IoU} > \tau_{iou} \text{ and } d_{ReID} < \tau_{reid} \\
\infty & \text{otherwise}
\end{cases}
$$

### 2.3 Assignment and Early Stopping
The optimal assignment is found by minimizing the total cost using the **Hungarian Algorithm** (Linear Sum Assignment). 
If a tracklet $T_i$ is not matched (due to occlusion or the strict gating thresholds), it is immediately marked as `Lost` and its propagation in the local phase is terminated. Unmatched detections instantiate new tracklets.

---

## 3. Phase 2: Global Association

In the offline phase, we take the entire set of tracklets (both active and lost) generated across the video, $\mathcal{T}_{all}$, and construct a directed graph where nodes are tracklets and edges represent feasible links across occlusions.

### 3.1 Spatio-Temporal Feasibility
We only compute edge weights between tracklet $T_A$ and tracklet $T_B$ if they are physically capable of representing the same target.
1. **Temporal Causality:** $T_B$ must begin after $T_A$ ends. Let $\Delta t = t_{start}(T_B) - t_{end}(T_A)$. We require $0 < \Delta t < \Delta t_{max}$.
2. **Spatial Kinematics:** The Euclidean distance $\Delta d$ between the center of the last bounding box of $T_A$ and the first bounding box of $T_B$ must not exceed a maximum physical velocity $v_{max}$. We require $\frac{\Delta d}{\Delta t} < v_{max}$.

### 3.2 Tracklet Affinity
If an edge $(A, B)$ is feasible, we evaluate the ReID affinity. Since tracklets contain multiple embeddings, we compare their smoothed representations (EMA). The edge cost $E_{A,B}$ is computed as:

$$
E_{A, B} = d_{ReID}(\mathbf{\bar{e}}_A, \mathbf{\bar{e}}_B) + \lambda \cdot \left( \frac{\Delta t}{\Delta t_{max}} \right)
$$

The temporal penalty term $\lambda$ gently biases the matching toward shorter temporal gaps to prevent illogical jumps when multiple targets share similar appearances.

### 3.3 Global Optimization and Interpolation
We construct a global cost matrix $\mathbf{E}$ and solve it using the Hungarian Algorithm to find the optimal tracklet-to-tracklet bipartite matching.
When $T_A$ is successfully linked to $T_B$, they are merged into a single trajectory under a global ID. 

Because $\Delta t > 1$, there will be missing bounding boxes in the timeline between $T_A$ and $T_B$. We resolve this by performing simple **Linear Interpolation** in the image plane for the coordinates of the bounding box across the unobserved frames, ensuring a smooth, continuous output trajectory for the downstream evaluation metrics.

---

## Usage

Run the offline tracker by specifying the root folder, the detection file, and the ReID model weights:

```bash
python tracker/offline_tracker.py \
    --seq_root "/path/to/dancetrack0094" \
    --det_path "/path/to/dancetrack0094.txt" \
    --weight "../pretrained/checkpoint0260.pth" \
    --out_path "tracking_results.txt" \
    --device "cuda"
```
