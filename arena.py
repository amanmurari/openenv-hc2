"""
Comparative Agent Arena

Run multiple agents simultaneously to compare performance.
Supports:
- Rule-based agent
- LLM agent (makes live API calls)
- Random agent
"""

import os
import json
import textwrap
import time
import random
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from traffic_control.environment import TrafficControlEnvironment
from traffic_control.models import TrafficAction, TrafficObservation
from traffic_control.tasks import grade, GradeResult


try:
    from openai import OpenAI
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


@dataclass
class AgentResult:
    """Result for a single agent run."""
    agent_name: str
    agent_type: str
    task_id: str
    episode_id: str
    steps: int = 0
    total_reward: float = 0.0
    score: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    grade_result: Optional[GradeResult] = None
    decision_times: List[float] = field(default_factory=list)
    
    @property
    def avg_decision_time_ms(self) -> float:
        if self.decision_times:
            return sum(self.decision_times) / len(self.decision_times) * 1000
        return 0.0


class RuleBasedAgent:
    """Rule-based traffic controller (optimized for high scores)."""
    
    def __init__(self, name: str = "RuleBased"):
        self.name = name
    
    def decide(self, obs: TrafficObservation) -> int:
        """Return light phase based on rules."""
        em_q = obs.emergency_queue
        em_u = obs.emergency_urgency
        q = obs.queue_lengths
        current = obs.current_phase
        time_in = obs.time_in_phase
        
        # Emergency prioritization
        ns_em_urgency = em_u[0] + em_u[1] + em_q[0] * 2 + em_q[1] * 2
        ew_em_urgency = em_u[2] + em_u[3] + em_q[2] * 2 + em_q[3] * 2
        
        if ns_em_urgency > 0 or ew_em_urgency > 0:
            return 0 if ns_em_urgency >= ew_em_urgency else 1
        
        # Queue-based switching
        ns_total = q[0] + q[1]
        ew_total = q[2] + q[3]
        min_phase_time = min(3 + max(ns_total, ew_total) // 5, 8)
        
        if current == 0 and time_in < min_phase_time and ns_total > 0:
            return 0
        if current == 1 and time_in < min_phase_time and ew_total > 0:
            return 1
        
        if ns_total >= ew_total + 2:
            return 0
        elif ew_total >= ns_total + 2:
            return 1
        else:
            return current if current in (0, 1) else 0


class RandomAgent:
    """Random traffic controller for baseline comparison."""
    
    def __init__(self, name: str = "Random", seed: int = 42):
        self.name = name
        self.rng = random.Random(seed)
    
    def decide(self, obs: TrafficObservation) -> int:
        """Return random light phase."""
        return self.rng.choice([0, 1, 2])


class RoundRobinAgent:
    """Simple round-robin controller."""
    
    def __init__(self, name: str = "RoundRobin", switch_interval: int = 5):
        self.name = name
        self.switch_interval = switch_interval
        self.step_count = 0
    
    def decide(self, obs: TrafficObservation) -> int:
        """Alternate between phases."""
        self.step_count += 1
        phase_index = (self.step_count // self.switch_interval) % 2
        return int(phase_index)


# LLM System Prompt for arena
LLM_SYSTEM_PROMPT = textwrap.dedent("""
    You are an Autonomous Traffic Control AI managing a 4-way intersection.

    PHASES:
      0 = North-South Green  (N/S vehicles may pass)
      1 = East-West Green    (E/W vehicles may pass)
      2 = All Red            (no vehicles pass)

    DECISION RULES (apply in order):
      1. EMERGENCY CHECK: If emergency vehicles are waiting, prioritize them.
      2. MINIMUM PHASE TIME: Stay in current phase at least 3 steps if traffic present.
      3. QUEUE BALANCE: Switch to direction with significantly more traffic.

    OUTPUT: Reply with exactly one JSON object: {"light_phase": <0, 1, or 2>}
""")


class LLM_Agent:
    """LLM-powered agent that makes dynamic API calls."""
    
    def __init__(
        self,
        name: str = "LLM-Agent",
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4.1-mini",
    ):
        self.name = name
        self.model_name = model_name
        self.api_calls_made = 0
        
        # Initialize OpenAI client if credentials available
        if _HAS_OPENAI and (api_base_url or api_key):
            self.client = OpenAI(
                base_url=api_base_url or "https://api.openai.com/v1",
                api_key=api_key or "dummy-key",
            )
        else:
            self.client = None
    
    def _build_prompt(self, obs: TrafficObservation) -> str:
        """Build the user prompt from observation."""
        return (
            f"Current phase: {obs.current_phase} (0=NS Green, 1=EW Green, 2=All Red)\n"
            f"Time in phase: {obs.time_in_phase} steps\n"
            f"\n"
            f"Queue lengths (N, S, E, W): {obs.queue_lengths}\n"
            f"Emergency queues (N, S, E, W): {obs.emergency_queue}\n"
            f"Emergency urgency (N, S, E, W): {obs.emergency_urgency}\n"
            f"\n"
            f"What light phase should be set? Respond with JSON: {{\"light_phase\": 0, 1, or 2}}"
        )
    
    def decide(self, obs: TrafficObservation) -> int:
        """Make LLM API call to get decision."""
        if not self.client:
            # Fallback to rule-based if no client
            return self._rule_fallback(obs)
        
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": self._build_prompt(obs)},
                ],
                temperature=0.0,
                max_tokens=32,
                stream=False,
            )
            self.api_calls_made += 1
            
            content = resp.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                data = json.loads(content)
                phase = int(data.get("light_phase", 0))
                return max(0, min(2, phase))  # Clamp to valid range
            except (json.JSONDecodeError, ValueError, KeyError):
                # Fallback if parsing fails
                return self._rule_fallback(obs)
                
        except Exception as e:
            # Fallback on API error
            print(f"[LLM Agent] API error: {e}, using fallback")
            return self._rule_fallback(obs)
    
    def _rule_fallback(self, obs: TrafficObservation) -> int:
        """Rule-based fallback when LLM fails."""
        em_q = obs.emergency_queue
        em_u = obs.emergency_urgency
        q = obs.queue_lengths
        current = obs.current_phase
        
        # Emergency prioritization
        ns_em = em_u[0] + em_u[1] + em_q[0] + em_q[1]
        ew_em = em_u[2] + em_u[3] + em_q[2] + em_q[3]
        
        if ns_em > 0 or ew_em > 0:
            return 0 if ns_em >= ew_em else 1
        
        # Queue-based
        ns_total = q[0] + q[1]
        ew_total = q[2] + q[3]
        
        if ns_total > ew_total:
            return 0
        elif ew_total > ns_total:
            return 1
        else:
            return current if current in (0, 1) else 0


class Arena:
    """Run multiple agents and compare results."""
    
    def __init__(self):
        self.results: List[AgentResult] = []
        
        # Get LLM credentials from env (for arena LLM agent)
        api_base = os.environ.get("API_BASE_URL")
        api_key = os.environ.get("API_KEY")
        model = os.environ.get("MODEL_NAME", "gpt-4.1-mini")
        
        self.agents = {
            "llm": LLM_Agent("Dynamic LLM", api_base, api_key, model),
            "rule_based": RuleBasedAgent("Smart Rule-Based"),
            "random": RandomAgent("Random Baseline"),
            "round_robin": RoundRobinAgent("Round Robin"),
        }
    
    async def run_agent(
        self,
        agent_type: str,
        task_id: str,
        max_steps: int = 300,
        seed: int = 42,
    ) -> AgentResult:
        """Run a single agent episode."""
        env = TrafficControlEnvironment(task_id=task_id)
        agent = self.agents.get(agent_type, self.agents["rule_based"])
        
        obs = env.reset(seed=seed)
        episode_id = env._episode_id
        
        result = AgentResult(
            agent_name=agent.name,
            agent_type=agent_type,
            task_id=task_id,
            episode_id=episode_id,
        )
        
        for step in range(max_steps):
            import time
            start_time = time.time()
            
            action_id = agent.decide(obs)
            action = TrafficAction(light_phase=action_id)
            
            decision_time = time.time() - start_time
            result.decision_times.append(decision_time)
            
            obs = env.step(action)
            result.steps = step + 1
            result.total_reward += obs.reward or 0.0
            
            if obs.done:
                break
        
        # Grade the result
        state = env.state
        result.grade_result = grade(
            task_id,
            total_vehicles_passed=state.total_vehicles_passed,
            total_emergency_passed=state.total_emergency_passed,
            total_waiting_time=state.total_waiting_time,
            total_collisions=state.total_collisions,
            total_emergency_delay=state.total_emergency_delay,
            total_phase_changes=state.total_phase_changes,
            step_count=result.steps,
        )
        result.score = result.grade_result.score
        result.metrics = result.grade_result.metrics
        
        return result
    
    async def run_comparison(
        self,
        task_id: str,
        agents: Optional[List[str]] = None,
        runs_per_agent: int = 1,
    ) -> Dict[str, Any]:
        """Run multiple agents and compare."""
        agents_to_run = agents or list(self.agents.keys())
        all_results = []
        
        for agent_type in agents_to_run:
            for run in range(runs_per_agent):
                seed = 42 + run
                result = await self.run_agent(agent_type, task_id, seed=seed)
                all_results.append(result)
        
        self.results.extend(all_results)
        
        # Aggregate results
        summary = self._aggregate_results(all_results)
        return summary
    
    def _aggregate_results(self, results: List[AgentResult]) -> Dict[str, Any]:
        """Aggregate results by agent type."""
        by_agent: Dict[str, List[AgentResult]] = {}
        for r in results:
            if r.agent_type not in by_agent:
                by_agent[r.agent_type] = []
            by_agent[r.agent_type].append(r)
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_runs": len(results),
            "agents": {},
            "winner": None,
        }
        
        best_score = -1
        best_agent = None
        
        for agent_type, agent_results in by_agent.items():
            avg_score = sum(r.score for r in agent_results) / len(agent_results)
            avg_reward = sum(r.total_reward for r in agent_results) / len(agent_results)
            avg_steps = sum(r.steps for r in agent_results) / len(agent_results)
            avg_time = sum(r.avg_decision_time_ms for r in agent_results) / len(agent_results)
            
            summary["agents"][agent_type] = {
                "name": agent_results[0].agent_name,
                "runs": len(agent_results),
                "avg_score": round(avg_score, 4),
                "avg_total_reward": round(avg_reward, 2),
                "avg_steps": round(avg_steps, 1),
                "avg_decision_time_ms": round(avg_time, 2),
                "best_run": max(agent_results, key=lambda r: r.score).episode_id,
            }
            
            if avg_score > best_score:
                best_score = avg_score
                best_agent = agent_type
        
        summary["winner"] = best_agent
        summary["all_runs"] = [
            {
                "agent": r.agent_type,
                "episode_id": r.episode_id,
                "score": r.score,
                "reward": round(r.total_reward, 2),
                "steps": r.steps,
            }
            for r in results
        ]
        
        return summary


# Global arena instance
_arena = Arena()


def get_arena() -> Arena:
    """Get the global arena instance."""
    return _arena
