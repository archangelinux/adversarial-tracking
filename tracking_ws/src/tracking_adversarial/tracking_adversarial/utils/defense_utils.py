"""Defense utilities for improving tracking robustness.

Two defense mechanisms:
1. TemporalConsistencyChecker — detects and corrects anomalous track jumps
   using motion prediction (weighted velocity averaging)
2. AnomalyDetector — flags suspicious motion patterns using z-score
   analysis on velocity and acceleration
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TrackState:
    """Internal state for a single tracked object."""
    positions: deque = field(default_factory=lambda: deque(maxlen=60))
    velocities: deque = field(default_factory=lambda: deque(maxlen=60))
    accelerations: deque = field(default_factory=lambda: deque(maxlen=60))
    bbox: np.ndarray = field(default_factory=lambda: np.zeros(4))
    predicted_position: Optional[np.ndarray] = None
    lost_frames: int = 0
    is_anomalous: bool = False


class TemporalConsistencyChecker:
    """Detects and corrects anomalous track position jumps.

    Maintains a history of positions for each track and uses Kalman-style
    motion prediction to detect impossible jumps. When a jump exceeds
    the threshold, the track position is replaced with the predicted
    position.
    """

    def __init__(
        self,
        max_position_jump: float = 100.0,
        history_length: int = 30,
        recovery_frames: int = 10,
    ) -> None:
        self.max_position_jump = max_position_jump
        self.history_length = history_length
        self.recovery_frames = recovery_frames
        self.tracks: Dict[int, TrackState] = {}

    def check_and_correct(
        self,
        track_id: int,
        bbox: np.ndarray,
    ) -> Tuple[np.ndarray, bool]:
        """Check if a track position is consistent and correct if not.

        Args:
            track_id: Unique track identifier.
            bbox: Current bounding box [cx, cy, w, h].

        Returns:
            Tuple of (corrected_bbox, was_corrected).
        """
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackState(
                positions=deque(maxlen=self.history_length),
                bbox=bbox.copy(),
            )
            self.tracks[track_id].positions.append(bbox[:2].copy())
            return bbox, False

        state = self.tracks[track_id]
        current_pos = bbox[:2]

        # Predict expected position from velocity history
        predicted_pos = self._predict_position(state)

        # Check for position jump
        was_corrected = False
        if predicted_pos is not None:
            jump_dist = np.linalg.norm(current_pos - predicted_pos)

            if jump_dist > self.max_position_jump:
                # Replace with predicted position, keep size
                bbox = bbox.copy()
                bbox[:2] = predicted_pos
                was_corrected = True
                state.lost_frames += 1
            else:
                state.lost_frames = 0

        # Update history
        state.positions.append(bbox[:2].copy())
        state.bbox = bbox.copy()

        # Update velocity
        if len(state.positions) >= 2:
            vel = state.positions[-1] - state.positions[-2]
            state.velocities.append(vel)

        return bbox, was_corrected

    def _predict_position(self, state: TrackState) -> Optional[np.ndarray]:
        """Predict next position using average velocity."""
        if len(state.velocities) < 2:
            return None

        # Weighted average of recent velocities (more recent = higher weight)
        vels = np.array(list(state.velocities))
        weights = np.linspace(0.5, 1.0, len(vels))
        avg_vel = np.average(vels, axis=0, weights=weights)

        return state.positions[-1] + avg_vel

    def recover_lost_tracks(
        self,
        active_ids: set,
    ) -> List[Tuple[int, np.ndarray]]:
        """Attempt to recover tracks that have been lost.

        Args:
            active_ids: Set of currently active track IDs.

        Returns:
            List of (track_id, predicted_bbox) for recoverable tracks.
        """
        recovered = []
        for tid, state in self.tracks.items():
            if tid in active_ids:
                continue
            if state.lost_frames > self.recovery_frames:
                continue

            predicted = self._predict_position(state)
            if predicted is not None:
                recovered_bbox = state.bbox.copy()
                recovered_bbox[:2] = predicted
                recovered.append((tid, recovered_bbox))
                state.lost_frames += 1
                state.positions.append(predicted)

        return recovered

    def cleanup_stale(self, active_ids: set) -> None:
        """Remove tracks that have been lost too long."""
        stale = [
            tid for tid, state in self.tracks.items()
            if tid not in active_ids and state.lost_frames > self.recovery_frames
        ]
        for tid in stale:
            del self.tracks[tid]


class AnomalyDetector:
    """Detects anomalous motion patterns in tracked objects.

    Uses z-score based anomaly detection on velocity and acceleration
    to flag tracks with suspicious behavior (e.g., impossible speed
    changes that may indicate ID switch or adversarial manipulation).
    """

    def __init__(
        self,
        velocity_threshold: float = 3.0,
        acceleration_threshold: float = 3.0,
        min_history: int = 5,
    ) -> None:
        self.velocity_threshold = velocity_threshold
        self.acceleration_threshold = acceleration_threshold
        self.min_history = min_history
        self.track_stats: Dict[int, TrackState] = {}

    def update(
        self,
        track_id: int,
        position: np.ndarray,
    ) -> Tuple[bool, float]:
        """Update motion statistics and check for anomaly.

        Args:
            track_id: Unique track identifier.
            position: Current position [cx, cy].

        Returns:
            Tuple of (is_anomalous, anomaly_score). Score is the max
            z-score across velocity and acceleration dimensions.
        """
        if track_id not in self.track_stats:
            self.track_stats[track_id] = TrackState()

        state = self.track_stats[track_id]
        state.positions.append(position.copy())

        # Compute velocity
        if len(state.positions) >= 2:
            vel = state.positions[-1] - state.positions[-2]
            state.velocities.append(vel)

        # Compute acceleration
        if len(state.velocities) >= 2:
            acc = np.array(state.velocities[-1]) - np.array(state.velocities[-2])
            state.accelerations.append(acc)

        # Need minimum history for meaningful statistics
        if len(state.velocities) < self.min_history:
            return False, 0.0

        # Z-score anomaly detection on velocity
        vels = np.array(list(state.velocities))
        vel_mean = np.mean(vels, axis=0)
        vel_std = np.std(vels, axis=0)
        vel_std = np.maximum(vel_std, 1e-6)  # Avoid division by zero

        current_vel = np.array(state.velocities[-1])
        vel_zscore = np.abs((current_vel - vel_mean) / vel_std)
        max_vel_z = float(np.max(vel_zscore))

        # Z-score on acceleration
        max_acc_z = 0.0
        if len(state.accelerations) >= self.min_history:
            accs = np.array(list(state.accelerations))
            acc_mean = np.mean(accs, axis=0)
            acc_std = np.std(accs, axis=0)
            acc_std = np.maximum(acc_std, 1e-6)

            current_acc = np.array(state.accelerations[-1])
            acc_zscore = np.abs((current_acc - acc_mean) / acc_std)
            max_acc_z = float(np.max(acc_zscore))

        # Combined anomaly score
        anomaly_score = max(max_vel_z, max_acc_z)
        is_anomalous = (
            max_vel_z > self.velocity_threshold
            or max_acc_z > self.acceleration_threshold
        )

        state.is_anomalous = is_anomalous
        return is_anomalous, anomaly_score

    def get_anomalous_tracks(self) -> List[int]:
        """Return list of currently anomalous track IDs."""
        return [
            tid for tid, state in self.track_stats.items()
            if state.is_anomalous
        ]

    def cleanup(self, active_ids: set) -> None:
        """Remove statistics for inactive tracks."""
        stale = [tid for tid in self.track_stats if tid not in active_ids]
        for tid in stale:
            del self.track_stats[tid]
