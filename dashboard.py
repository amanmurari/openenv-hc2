"""
Real-time Visual Dashboard for Traffic Control Environment.

Generates SVG visualizations of the 4-way intersection showing:
- Vehicles in each queue
- Traffic light states with color-coded signals
- Emergency vehicles with flashing indicators
- Live statistics overlay
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import json


@dataclass
class RenderState:
    """Current state for rendering."""
    current_phase: int
    time_in_phase: int
    queue_lengths: List[int]
    emergency_queue: List[int]
    emergency_urgency: List[int]
    total_vehicles_passed: int = 0
    total_emergency_passed: int = 0
    step_count: int = 0
    reward: float = 0.0


def get_phase_color(phase: int, direction: str) -> str:
    """Get traffic light color for a given phase and direction."""
    colors = {
        0: {"NS": "#22c55e", "EW": "#ef4444"},  # NS_GREEN
        1: {"NS": "#ef4444", "EW": "#22c55e"},  # EW_GREEN
        2: {"NS": "#ef4444", "EW": "#ef4444"},  # ALL_RED
        3: {"NS": "#eab308", "EW": "#ef4444"},  # NS_YELLOW
        4: {"NS": "#ef4444", "EW": "#eab308"},  # EW_YELLOW
    }
    mapping = colors.get(phase, colors[0])
    return mapping.get(direction, "#ef4444")


def render_intersection(state: RenderState, width: int = 600, height: int = 600) -> str:
    """
    Render the intersection as an SVG string.
    
    Args:
        state: Current simulation state
        width: SVG width in pixels
        height: SVG height in pixels
        
    Returns:
        SVG string
    """
    cx, cy = width // 2, height // 2
    road_width = 80
    lane_width = road_width // 2
    
    # Colors
    bg_color = "#1a1a2e"
    road_color = "#2d2d44"
    line_color = "#fbbf24"
    text_color = "#e2e8f0"
    
    svg_parts = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{width}" height="{height}" fill="{bg_color}"/>',
    ]
    
    # Draw roads
    # Vertical road (North-South)
    svg_parts.append(
        f'<rect x="{cx - road_width//2}" y="0" width="{road_width}" height="{height}" fill="{road_color}"/>'
    )
    # Horizontal road (East-West)
    svg_parts.append(
        f'<rect x="0" y="{cy - road_width//2}" width="{width}" height="{road_width}" fill="{road_color}"/>'
    )
    
    # Draw center intersection
    svg_parts.append(
        f'<rect x="{cx - road_width//2}" y="{cy - road_width//2}" width="{road_width}" height="{road_width}" fill="{road_color}"/>'
    )
    
    # Draw lane markings
    dash_length = 20
    gap_length = 20
    for y in range(0, height, dash_length + gap_length):
        svg_parts.append(
            f'<rect x="{cx - 2}" y="{y}" width="4" height="{dash_length}" fill="{line_color}" opacity="0.5"/>'
        )
    for x in range(0, width, dash_length + gap_length):
        svg_parts.append(
            f'<rect x="{x}" y="{cy - 2}" width="{dash_length}" height="4" fill="{line_color}" opacity="0.5"/>'
        )
    
    # Draw stop lines
    stop_line_offset = road_width // 2 + 10
    svg_parts.extend([
        # North stop line
        f'<line x1="{cx - road_width//2}" y1="{cy - stop_line_offset}" x2="{cx + road_width//2}" y2="{cy - stop_line_offset}" stroke="{line_color}" stroke-width="3"/>',
        # South stop line
        f'<line x1="{cx - road_width//2}" y1="{cy + stop_line_offset}" x2="{cx + road_width//2}" y2="{cy + stop_line_offset}" stroke="{line_color}" stroke-width="3"/>',
        # East stop line
        f'<line x1="{cx + stop_line_offset}" y1="{cy - road_width//2}" x2="{cx + stop_line_offset}" y2="{cy + road_width//2}" stroke="{line_color}" stroke-width="3"/>',
        # West stop line
        f'<line x1="{cx - stop_line_offset}" y1="{cy - road_width//2}" x2="{cx - stop_line_offset}" y2="{cy + road_width//2}" stroke="{line_color}" stroke-width="3"/>',
    ])
    
    # Draw traffic lights
    light_radius = 12
    ns_color = get_phase_color(state.current_phase, "NS")
    ew_color = get_phase_color(state.current_phase, "EW")
    
    # North light
    svg_parts.append(
        f'<circle cx="{cx + road_width//2 + 20}" cy="{cy - stop_line_offset}" r="{light_radius}" fill="{ns_color}" stroke="white" stroke-width="2"/>'
    )
    # South light
    svg_parts.append(
        f'<circle cx="{cx - road_width//2 - 20}" cy="{cy + stop_line_offset}" r="{light_radius}" fill="{ns_color}" stroke="white" stroke-width="2"/>'
    )
    # East light
    svg_parts.append(
        f'<circle cx="{cx + stop_line_offset}" cy="{cy + road_width//2 + 20}" r="{light_radius}" fill="{ew_color}" stroke="white" stroke-width="2"/>'
    )
    # West light
    svg_parts.append(
        f'<circle cx="{cx - stop_line_offset}" cy="{cy - road_width//2 - 20}" r="{light_radius}" fill="{ew_color}" stroke="white" stroke-width="2"/>'
    )
    
    # Draw vehicles in queues
    car_width = 24
    car_height = 36
    truck_width = 28
    truck_height = 44
    
    def draw_vehicle(x: float, y: float, is_emergency: bool, urgency: int, rotation: int = 0):
        if is_emergency:
            # Emergency vehicle (flashing red/blue)
            flash = "#ef4444" if state.step_count % 4 < 2 else "#3b82f6"
            return (
                f'<g transform="translate({x},{y}) rotate({rotation})">'
                f'<rect x="{-truck_width//2}" y="{-truck_height//2}" width="{truck_width}" height="{truck_height}" rx="4" fill="{flash}" stroke="white" stroke-width="2"/>'
                f'<text x="0" y="4" text-anchor="middle" fill="white" font-size="10" font-weight="bold">🚨</text>'
                f'<text x="0" y="{-truck_height//2 - 8}" text-anchor="middle" fill="#ef4444" font-size="12" font-weight="bold">!{urgency}</text>'
                f'</g>'
            )
        else:
            # Regular car
            car_colors = ["#60a5fa", "#34d399", "#f472b6", "#fbbf24"]
            color = car_colors[(int(x) + int(y)) % len(car_colors)]
            return (
                f'<g transform="translate({x},{y}) rotate({rotation})">'
                f'<rect x="{-car_width//2}" y="{-car_height//2}" width="{car_width}" height="{car_height}" rx="3" fill="{color}" stroke="white" stroke-width="1"/>'
                f'<rect x="{-car_width//2 + 4}" y="{-car_height//2 + 4}" width="{car_width - 8}" height="{car_height//3}" rx="2" fill="#1e293b"/>'
                f'</g>'
            )
    
    # Draw North queue (approaching from top)
    nx = cx - lane_width // 2
    for i in range(min(state.queue_lengths[0], 8)):
        y = 30 + i * 45
        svg_parts.append(draw_vehicle(nx, y, False, 0, 180))
    for i in range(min(state.emergency_queue[0], 2)):
        y = 30 + (state.queue_lengths[0] + i) * 45
        urgency = state.emergency_urgency[0] if i == 0 else 5
        svg_parts.append(draw_vehicle(nx, y, True, urgency, 180))
    
    # Draw South queue (approaching from bottom)
    sx = cx + lane_width // 2
    for i in range(min(state.queue_lengths[1], 8)):
        y = height - 30 - i * 45
        svg_parts.append(draw_vehicle(sx, y, False, 0, 0))
    for i in range(min(state.emergency_queue[1], 2)):
        y = height - 30 - (state.queue_lengths[1] + i) * 45
        urgency = state.emergency_urgency[1] if i == 0 else 5
        svg_parts.append(draw_vehicle(sx, y, True, urgency, 0))
    
    # Draw East queue (approaching from right)
    ey = cy + lane_width // 2
    for i in range(min(state.queue_lengths[2], 8)):
        x = width - 30 - i * 45
        svg_parts.append(draw_vehicle(x, ey, False, 0, 270))
    for i in range(min(state.emergency_queue[2], 2)):
        x = width - 30 - (state.queue_lengths[2] + i) * 45
        urgency = state.emergency_urgency[2] if i == 0 else 5
        svg_parts.append(draw_vehicle(x, ey, True, urgency, 270))
    
    # Draw West queue (approaching from left)
    wy = cy - lane_width // 2
    for i in range(min(state.queue_lengths[3], 8)):
        x = 30 + i * 45
        svg_parts.append(draw_vehicle(x, wy, False, 0, 90))
    for i in range(min(state.emergency_queue[3], 2)):
        x = 30 + (state.queue_lengths[3] + i) * 45
        urgency = state.emergency_urgency[3] if i == 0 else 5
        svg_parts.append(draw_vehicle(x, wy, True, urgency, 90))
    
    # Draw labels
    svg_parts.extend([
        f'<text x="{width//2}" y="25" text-anchor="middle" fill="{text_color}" font-size="18" font-weight="bold">🚦 Traffic Control Dashboard</text>',
        f'<text x="15" y="{height//2 - 60}" fill="{text_color}" font-size="14" transform="rotate(-90, 15, {height//2})">West ({state.queue_lengths[3]}🚗 {state.emergency_queue[3]}🚨)</text>',
        f'<text x="{width - 15}" y="{height//2 - 60}" fill="{text_color}" font-size="14" transform="rotate(90, {width - 15}, {height//2})">East ({state.queue_lengths[2]}🚗 {state.emergency_queue[2]}🚨)</text>',
        f'<text x="{width//2}" y="{height - 10}" text-anchor="middle" fill="{text_color}" font-size="14">South ({state.queue_lengths[1]}🚗 {state.emergency_queue[1]}🚨)</text>',
        f'<text x="{width//2}" y="{height - 30}" text-anchor="middle" fill="{text_color}" font-size="14">North ({state.queue_lengths[0]}🚗 {state.emergency_queue[0]}🚨)</text>',
    ])
    
    # Draw stats panel
    stats_x = 10
    stats_y = height - 100
    svg_parts.append(
        f'<rect x="{stats_x}" y="{stats_y}" width="180" height="90" rx="8" fill="rgba(0,0,0,0.5)" stroke="{text_color}" stroke-width="1"/>'
    )
    
    phase_names = {0: "NS GREEN", 1: "EW GREEN", 2: "ALL RED", 3: "NS YELLOW", 4: "EW YELLOW"}
    phase_name = phase_names.get(state.current_phase, "UNKNOWN")
    
    stats_text = [
        f"Step: {state.step_count}",
        f"Phase: {phase_name} ({state.time_in_phase}s)",
        f"Reward: {state.reward:.2f}",
        f"Passed: {state.total_vehicles_passed}🚗 {state.total_emergency_passed}🚨",
    ]
    
    for i, line in enumerate(stats_text):
        svg_parts.append(
            f'<text x="{stats_x + 10}" y="{stats_y + 20 + i * 20}" fill="{text_color}" font-size="12" font-family="monospace">{line}</text>'
        )
    
    svg_parts.append('</svg>')
    return '\n'.join(svg_parts)


def observation_to_render_state(
    obs: Dict[str, Any],
    total_vehicles: int = 0,
    total_emergency: int = 0,
    step: int = 0,
    reward: float = 0.0,
) -> RenderState:
    """Convert observation dict to RenderState."""
    return RenderState(
        current_phase=obs.get("current_phase", 0),
        time_in_phase=obs.get("time_in_phase", 0),
        queue_lengths=obs.get("queue_lengths", [0, 0, 0, 0]),
        emergency_queue=obs.get("emergency_queue", [0, 0, 0, 0]),
        emergency_urgency=obs.get("emergency_urgency", [0, 0, 0, 0]),
        total_vehicles_passed=total_vehicles,
        total_emergency_passed=total_emergency,
        step_count=step,
        reward=reward,
    )
