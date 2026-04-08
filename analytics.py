"""
Performance Analytics & Episode History

Tracks episode metrics over time and provides analysis tools.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import os


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    episode_id: str
    task_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    steps: int = 0
    total_reward: float = 0.0
    total_vehicles: int = 0
    total_emergency: int = 0
    total_waiting_time: float = 0.0
    total_collisions: int = 0
    phase_changes: int = 0
    avg_queue_length: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    rewards_history: List[float] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def duration_seconds(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def throughput_per_step(self) -> float:
        if self.steps > 0:
            return self.total_vehicles / self.steps
        return 0.0
    
    @property
    def avg_reward_per_step(self) -> float:
        if self.steps > 0:
            return self.total_reward / self.steps
        return 0.0


class EpisodeHistory:
    """Store and analyze episode history."""
    
    def __init__(self, max_episodes: int = 100):
        self.episodes: List[EpisodeMetrics] = []
        self.max_episodes = max_episodes
        self._current: Optional[EpisodeMetrics] = None
    
    def start_episode(self, episode_id: str, task_id: str) -> EpisodeMetrics:
        """Start tracking a new episode."""
        episode = EpisodeMetrics(
            episode_id=episode_id,
            task_id=task_id,
            start_time=datetime.now(),
        )
        self._current = episode
        return episode
    
    def record_step(
        self,
        step: int,
        reward: float,
        action: Dict[str, Any],
        observation: Dict[str, Any],
    ) -> None:
        """Record a step in the current episode."""
        if self._current:
            self._current.steps = step
            self._current.total_reward += reward
            self._current.rewards_history.append(reward)
            
            # Track queue lengths for averaging
            queues = observation.get("queue_lengths", [0, 0, 0, 0])
            for i, q in enumerate(queues):
                self._current.avg_queue_length[i] = (
                    self._current.avg_queue_length[i] * (step - 1) + q
                ) / step
            
            # Record decision
            self._current.decisions.append({
                "step": step,
                "action": action,
                "phase": observation.get("current_phase"),
                "queues": queues,
                "emergency_queues": observation.get("emergency_queue", [0, 0, 0, 0]),
            })
    
    def record_state(self, state: Dict[str, Any]) -> None:
        """Record final state metrics."""
        if self._current:
            self._current.total_vehicles = state.get("total_vehicles_passed", 0)
            self._current.total_emergency = state.get("total_emergency_passed", 0)
            self._current.total_waiting_time = state.get("total_waiting_time", 0.0)
            self._current.total_collisions = state.get("total_collisions", 0)
            self._current.phase_changes = state.get("total_phase_changes", 0)
    
    def end_episode(self) -> EpisodeMetrics:
        """Finalize the current episode."""
        if self._current:
            self._current.end_time = datetime.now()
            self.episodes.append(self._current)
            
            # Trim old episodes
            if len(self.episodes) > self.max_episodes:
                self.episodes = self.episodes[-self.max_episodes:]
            
            result = self._current
            self._current = None
            return result
        
        raise RuntimeError("No active episode to end")
    
    def get_summary(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Get summary statistics."""
        episodes = self.episodes
        if task_id:
            episodes = [e for e in episodes if e.task_id == task_id]
        
        if not episodes:
            return {"message": "No episodes recorded yet"}
        
        total_episodes = len(episodes)
        avg_reward = sum(e.avg_reward_per_step for e in episodes) / total_episodes
        avg_throughput = sum(e.throughput_per_step for e in episodes) / total_episodes
        avg_duration = sum(e.duration_seconds for e in episodes) / total_episodes
        
        # Find best episode
        best_idx = max(range(total_episodes), key=lambda i: episodes[i].avg_reward_per_step)
        best = episodes[best_idx]
        
        return {
            "total_episodes": total_episodes,
            "avg_reward_per_step": round(avg_reward, 4),
            "avg_throughput_per_step": round(avg_throughput, 4),
            "avg_duration_seconds": round(avg_duration, 2),
            "best_episode": {
                "episode_id": best.episode_id,
                "task_id": best.task_id,
                "reward_per_step": round(best.avg_reward_per_step, 4),
                "total_reward": round(best.total_reward, 2),
                "steps": best.steps,
            },
            "recent_performance": [
                {
                    "episode_id": e.episode_id,
                    "task_id": e.task_id,
                    "reward_per_step": round(e.avg_reward_per_step, 4),
                    "steps": e.steps,
                }
                for e in episodes[-10:]
            ],
        }
    
    def get_episode_details(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed metrics for a specific episode."""
        for episode in self.episodes:
            if episode.episode_id == episode_id:
                return {
                    "episode_id": episode.episode_id,
                    "task_id": episode.task_id,
                    "start_time": episode.start_time.isoformat(),
                    "end_time": episode.end_time.isoformat() if episode.end_time else None,
                    "duration_seconds": episode.duration_seconds,
                    "steps": episode.steps,
                    "total_reward": round(episode.total_reward, 2),
                    "total_vehicles": episode.total_vehicles,
                    "total_emergency": episode.total_emergency,
                    "total_collisions": episode.total_collisions,
                    "phase_changes": episode.phase_changes,
                    "avg_queue_lengths": [round(q, 2) for q in episode.avg_queue_length],
                    "throughput_per_step": round(episode.throughput_per_step, 4),
                    "reward_per_step": round(episode.avg_reward_per_step, 4),
                    "rewards_history": [round(r, 3) for r in episode.rewards_history],
                    "decision_count": len(episode.decisions),
                }
        return None
    
    def export_to_json(self, filepath: str) -> None:
        """Export all episodes to JSON file."""
        data = {
            "export_time": datetime.now().isoformat(),
            "total_episodes": len(self.episodes),
            "episodes": [
                {
                    "episode_id": e.episode_id,
                    "task_id": e.task_id,
                    "steps": e.steps,
                    "total_reward": e.total_reward,
                    "total_vehicles": e.total_vehicles,
                    "avg_reward_per_step": e.avg_reward_per_step,
                }
                for e in self.episodes
            ],
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# Global history instance
_episode_history = EpisodeHistory()


def get_history() -> EpisodeHistory:
    """Get the global episode history instance."""
    return _episode_history
