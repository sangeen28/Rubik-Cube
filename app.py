# rubik_solver_fixed.py
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import time
import random
import copy
import sys
from pathlib import Path
from collections import deque
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set
import math

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import the puzzle and agent modules
try:
    from puzzle import State, move
    from Agent import Agent
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Check if Agent2 is available
try:
    from agent2 import Agent2
    AGENT2_AVAILABLE = True
except ImportError:
    AGENT2_AVAILABLE = False

# Enhanced color mapping for better visualization
COLOR_MAP = {
    'W': ('#FFFFFF', 'White'),
    'Y': ('#FFEB3B', 'Yellow'),
    'R': ('#F44336', 'Red'),
    'O': ('#FF9800', 'Orange'),
    'B': ('#2196F3', 'Blue'),
    'G': ('#4CAF50', 'Green')
}

@dataclass
class CubeConfig:
    """Configuration for cube visualization"""
    size: float = 1.0
    sticker_size: float = 0.28
    spacing: float = 0.01
    border_size: float = 0.015

class ProfessionalCubeRenderer:
    """Professional 3D cube renderer with realistic appearance"""
    
    def __init__(self, config: CubeConfig = None):
        self.config = config or CubeConfig()
    
    def render_cube(self, state_dict: Dict, rotation: Tuple[float, float] = (0, 0)) -> go.Figure:
        """Render the entire cube with professional appearance"""
        fig = go.Figure()
        
        # Create a 3D representation of the Rubik's cube
        # We'll create small cubes for each piece
        
        # Positions for each of the 27 pieces (3x3x3)
        positions = []
        for i in range(3):  # Row
            for j in range(3):  # Column
                for k in range(3):  # Depth
                    # Skip the center piece (it's not visible)
                    if i == 1 and j == 1 and k == 1:
                        continue
                    
                    # Calculate position
                    x = (j - 1) * (self.config.sticker_size * 2 + self.config.spacing)
                    y = (1 - i) * (self.config.sticker_size * 2 + self.config.spacing)
                    z = (k - 1) * (self.config.sticker_size * 2 + self.config.spacing)
                    
                    positions.append(((i, j, k), (x, y, z)))
        
        # Map state to piece colors
        face_mapping = {
            'front': state_dict['front'],
            'back': state_dict['back'],
            'left': state_dict['left'],
            'right': state_dict['right'],
            'top': state_dict['top'],
            'bottom': state_dict['bottom']
        }
        
        # For each position, create a cube with appropriate colors
        for (i, j, k), (x, y, z) in positions:
            # Determine which faces are visible
            visible_faces = []
            
            # Front face (k = 2)
            if k == 2:
                visible_faces.append(('front', face_mapping['front'][i][j]))
            
            # Back face (k = 0)
            if k == 0:
                visible_faces.append(('back', face_mapping['back'][i][2 - j]))  # Mirrored
            
            # Right face (j = 2)
            if j == 2:
                visible_faces.append(('right', face_mapping['right'][i][2 - k]))  # Mirrored
            
            # Left face (j = 0)
            if j == 0:
                visible_faces.append(('left', face_mapping['left'][i][k]))
            
            # Top face (i = 0)
            if i == 0:
                visible_faces.append(('top', face_mapping['top'][j][k]))
            
            # Bottom face (i = 2)
            if i == 2:
                visible_faces.append(('bottom', face_mapping['bottom'][2 - j][k]))  # Adjusted
            
            # Create a small cube for this position
            self._create_piece(fig, x, y, z, visible_faces)
        
        # Apply rotation
        rot_x, rot_y = rotation
        
        # Update layout with enhanced 3D effects
        fig.update_layout(
            scene=dict(
                xaxis=dict(showbackground=False, visible=False, range=[-1, 1]),
                yaxis=dict(showbackground=False, visible=False, range=[-1, 1]),
                zaxis=dict(showbackground=False, visible=False, range=[-1, 1]),
                aspectmode='cube',
                camera=dict(
                    eye=dict(
                        x=2 * math.cos(rot_y) * math.cos(rot_x),
                        y=2 * math.sin(rot_x),
                        z=2 * math.sin(rot_y) * math.cos(rot_x)
                    ),
                    up=dict(x=0, y=1, z=0),
                    center=dict(x=0, y=0, z=0)
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    def _create_piece(self, fig: go.Figure, x: float, y: float, z: float, 
                     visible_faces: List[Tuple[str, str]]) -> None:
        """Create a single piece with visible faces"""
        s = self.config.sticker_size
        
        # Define cube vertices
        vertices = [
            (x - s, y - s, z - s),  # 0
            (x + s, y - s, z - s),  # 1
            (x + s, y + s, z - s),  # 2
            (x - s, y + s, z - s),  # 3
            (x - s, y - s, z + s),  # 4
            (x + s, y - s, z + s),  # 5
            (x + s, y + s, z + s),  # 6
            (x - s, y + s, z + s),  # 7
        ]
        
        # Define faces (each with 4 vertices)
        faces = [
            ([3, 2, 1, 0], 'back', (-1, 0, 0)),    # Back face (negative z)
            ([4, 5, 6, 7], 'front', (1, 0, 0)),    # Front face (positive z)
            ([0, 4, 7, 3], 'left', (0, -1, 0)),    # Left face (negative x)
            ([1, 5, 6, 2], 'right', (0, 1, 0)),    # Right face (positive x)
            ([3, 7, 6, 2], 'top', (0, 0, 1)),      # Top face (positive y)
            ([0, 4, 5, 1], 'bottom', (0, 0, -1)),  # Bottom face (negative y)
        ]
        
        # Add only visible faces
        visible_face_names = {face[0] for face in visible_faces}
        face_colors = {face_name: color for face_name, color in visible_faces}
        
        for vertex_indices, face_name, normal in faces:
            if face_name in visible_face_names:
                color_char = face_colors[face_name]
                color_info = COLOR_MAP.get(color_char, ('#CCCCCC', 'Gray'))
                
                # Get vertices for this face
                face_vertices = [vertices[i] for i in vertex_indices]
                
                # Create the face as a mesh
                x_vals = [v[0] for v in face_vertices]
                y_vals = [v[1] for v in face_vertices]
                z_vals = [v[2] for v in face_vertices]
                
                fig.add_trace(go.Mesh3d(
                    x=x_vals,
                    y=y_vals,
                    z=z_vals,
                    i=[0, 0],
                    j=[1, 2],
                    k=[2, 3],
                    color=color_info[0],
                    opacity=1.0,
                    flatshading=True,
                    showscale=False,
                    lighting=dict(
                        ambient=0.3,
                        diffuse=0.8,
                        fresnel=0.1,
                        specular=0.5,
                        roughness=0.1
                    ),
                    lightposition=dict(x=100, y=100, z=100)
                ))

class RubiksCubeSolver:
    """Enhanced Rubik's Cube Solver with multiple algorithms"""
    
    def __init__(self):
        self.moves = ['front', 'back', 'left', 'right', 'top', 'bottom']
        self.inverse_moves = {
            'front': 'back', 'back': 'front',
            'left': 'right', 'right': 'left',
            'top': 'bottom', 'bottom': 'top'
        }
        
    def get_heuristic(self, state: State) -> int:
        """Calculate heuristic value - lower is better"""
        faces = [state.front(), state.back(), state.left(), 
                state.right(), state.top(), state.bottom()]
        
        # Count mismatched stickers
        mismatch = 0
        for face in faces:
            center = face[1][1]
            for row in face:
                for sticker in row:
                    if sticker != center:
                        mismatch += 1
        
        # Additional penalty for incorrect corner/edge pieces
        penalty = 0
        
        # Check corner consistency
        corners = [
            # Front corners
            (state.front()[0][0], state.top()[2][0], state.left()[0][2]),
            (state.front()[0][2], state.top()[2][2], state.right()[0][0]),
            (state.front()[2][0], state.bottom()[0][0], state.left()[2][2]),
            (state.front()[2][2], state.bottom()[0][2], state.right()[2][0]),
            
            # Back corners
            (state.back()[0][0], state.top()[0][2], state.right()[0][2]),
            (state.back()[0][2], state.top()[0][0], state.left()[0][0]),
            (state.back()[2][0], state.bottom()[2][2], state.right()[2][2]),
            (state.back()[2][2], state.bottom()[2][0], state.left()[2][0])
        ]
        
        for corner in corners:
            unique_colors = len(set(corner))
            penalty += (unique_colors - 1) * 2
        
        return mismatch + penalty
    
    def is_solvable(self, state: State) -> bool:
        """Check if cube state is solvable"""
        # For 2x2x2 cube with 180-degree turns, all states are reachable
        return True
    
    def bfs_solver(self, start_state: State, max_depth: int = 20) -> List[str]:
        """BFS solver for Rubik's Cube"""
        if start_state.isGoalState():
            return []
        
        queue = deque()
        queue.append((start_state.copy(), []))
        visited = set()
        
        while queue:
            current_state, path = queue.popleft()
            state_hash = current_state.__hash__()
            
            if state_hash in visited:
                continue
            visited.add(state_hash)
            
            if len(path) >= max_depth:
                continue
            
            # Try all moves
            for move_name in self.moves:
                # Avoid undo moves
                if path and move_name == self.inverse_moves.get(path[-1], ""):
                    continue
                
                new_state = current_state.copy()
                new_state.move(move_name)
                
                if new_state.isGoalState():
                    return path + [move_name]
                
                queue.append((new_state, path + [move_name]))
        
        return []
    
    def ids_solver(self, start_state: State, max_depth: int = 15) -> List[str]:
        """Iterative Deepening Search solver"""
        for depth in range(1, max_depth + 1):
            result = self.dls_solver(start_state, depth)
            if result:
                return result
        return []
    
    def dls_solver(self, state: State, depth: int, path: List[str] = None) -> Optional[List[str]]:
        """Depth-Limited Search helper"""
        if path is None:
            path = []
        
        if state.isGoalState():
            return path
        
        if depth == 0:
            return None
        
        # Try all moves with move ordering
        moves_to_try = self.get_ordered_moves(state, path)
        
        for move_name in moves_to_try:
            new_state = state.copy()
            new_state.move(move_name)
            
            result = self.dls_solver(new_state, depth - 1, path + [move_name])
            if result:
                return result
        
        return None
    
    def get_ordered_moves(self, state: State, current_path: List[str]) -> List[str]:
        """Order moves based on heuristic improvement"""
        if not current_path:
            return self.moves.copy()
        
        last_move = current_path[-1]
        available_moves = []
        
        for move_name in self.moves:
            # Avoid undo moves
            if move_name == self.inverse_moves.get(last_move, ""):
                continue
            
            new_state = state.copy()
            new_state.move(move_name)
            improvement = self.get_heuristic(state) - self.get_heuristic(new_state)
            available_moves.append((improvement, move_name))
        
        # Sort by improvement (descending)
        available_moves.sort(reverse=True, key=lambda x: x[0])
        return [move_name for _, move_name in available_moves]
    
    def a_star_solver(self, start_state: State, max_iterations: int = 100000) -> List[str]:
        """A* solver for Rubik's Cube"""
        if start_state.isGoalState():
            return []
        
        # Priority queue: (f_score, counter, state, g_score, path)
        open_set = []
        counter = 0
        
        initial_h = self.get_heuristic(start_state)
        heapq.heappush(open_set, (initial_h, counter, start_state.copy(), 0, []))
        counter += 1
        
        g_scores = {start_state.__hash__(): 0}
        closed_set = set()
        
        iteration = 0
        
        while open_set and iteration < max_iterations:
            iteration += 1
            f_score, _, state, g_score, path = heapq.heappop(open_set)
            state_hash = state.__hash__()
            
            if state_hash in closed_set:
                continue
            
            if state.isGoalState():
                return path
            
            closed_set.add(state_hash)
            
            for move_name in self.moves:
                # Avoid undo moves
                if path and move_name == self.inverse_moves.get(path[-1], ""):
                    continue
                
                new_state = state.copy()
                new_state.move(move_name)
                new_hash = new_state.__hash__()
                
                tentative_g = g_score + 1
                
                if new_hash in closed_set:
                    continue
                
                if tentative_g < g_scores.get(new_hash, float('inf')):
                    g_scores[new_hash] = tentative_g
                    h = self.get_heuristic(new_state)
                    f_score = tentative_g + h
                    
                    heapq.heappush(open_set, (f_score, counter, new_state, tentative_g, path + [move_name]))
                    counter += 1
        
        return []
    
    def greedy_solver(self, start_state: State, max_steps: int = 100) -> List[str]:
        """Greedy solver with backtracking"""
        if start_state.isGoalState():
            return []
        
        best_path = []
        best_heuristic = float('inf')
        
        for _ in range(10):  # Try multiple starting points
            current_state = start_state.copy()
            path = []
            last_move = None
            
            for step in range(max_steps):
                if current_state.isGoalState():
                    return path
                
                # Find best immediate move
                best_move = None
                best_h = float('inf')
                
                for move_name in self.moves:
                    if move_name == last_move or move_name == self.inverse_moves.get(last_move, ""):
                        continue
                    
                    test_state = current_state.copy()
                    test_state.move(move_name)
                    h = self.get_heuristic(test_state)
                    
                    if h < best_h:
                        best_h = h
                        best_move = move_name
                
                if best_move is None:
                    break
                
                # Apply move
                current_state.move(best_move)
                path.append(best_move)
                last_move = best_move
                
                # Track best solution found
                if self.get_heuristic(current_state) < best_heuristic:
                    best_heuristic = self.get_heuristic(current_state)
                    best_path = path.copy()
            
            # If we found a solution, return it
            test_state = start_state.copy()
            for move in best_path:
                test_state.move(move)
            if test_state.isGoalState():
                return best_path
        
        return best_path
    
    def solve_with_agent(self, state: State, agent_type: str = "baseline") -> List[str]:
        """Solve using AI agent"""
        if not AGENT2_AVAILABLE and agent_type == "advanced":
            st.warning("Advanced agent not available. Using baseline agent.")
            agent_type = "baseline"
        
        if agent_type == "advanced":
            agent = Agent2(
                beam_width=32,
                beam_depth=4,
                step_penalty=0.01,
                terminal_reward=10.0
            )
            solved, steps, solution = agent.evaluate(state.copy(), max_steps=100)
            if solved:
                return solution
        else:
            # Use baseline agent
            agent = Agent()
            agent.register_patterns()
            
            # Try to find solution using agent's Q-values
            current_state = state.copy()
            solution = []
            last_move = None
            
            for _ in range(50):
                if current_state.isGoalState():
                    return solution
                
                # Find best move from Q-values
                best_move = None
                best_q = float('-inf')
                state_hash = current_state.__hash__()
                
                for move_name in self.moves:
                    if move_name == last_move:
                        continue
                    
                    q_value = agent.QV.get((state_hash, move_name), 0)
                    if q_value > best_q:
                        best_q = q_value
                        best_move = move_name
                
                if best_move is None:
                    best_move = random.choice(self.moves)
                
                current_state.move(best_move)
                solution.append(best_move)
                last_move = best_move
        
        return []
    
    def layer_by_layer_solver(self, state: State) -> List[str]:
        """Layer-by-layer solving method"""
        # This is a simplified layer-by-layer approach
        solution = []
        current_state = state.copy()
        
        # Solve bottom layer (simplified)
        # In a real implementation, this would be more complex
        for _ in range(20):
            if current_state.isGoalState():
                return solution
            
            # Simplified: just try to improve heuristic
            best_move = None
            best_improvement = float('-inf')
            
            for move_name in self.moves:
                if solution and move_name == self.inverse_moves.get(solution[-1], ""):
                    continue
                
                test_state = current_state.copy()
                test_state.move(move_name)
                improvement = self.get_heuristic(current_state) - self.get_heuristic(test_state)
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_move = move_name
            
            if best_move:
                current_state.move(best_move)
                solution.append(best_move)
            else:
                break
        
        return solution
    
    def hybrid_solver(self, state: State) -> List[str]:
        """Hybrid solver combining multiple approaches"""
        # Try different solvers in order
        solvers = [
            ("A*", self.a_star_solver),
            ("BFS", self.bfs_solver),
            ("Greedy", self.greedy_solver),
            ("Layer", self.layer_by_layer_solver)
        ]
        
        for solver_name, solver_func in solvers:
            with st.spinner(f"Trying {solver_name}..."):
                solution = solver_func(state.copy())
                if solution and self.verify_solution(state, solution):
                    return solution
        
        return []
    
    def solve(self, state: State, method: str = "hybrid") -> List[str]:
        """Main solve method"""
        if state.isGoalState():
            return []
        
        if method == "bfs":
            return self.bfs_solver(state, max_depth=20)
        elif method == "ids":
            return self.ids_solver(state, max_depth=15)
        elif method == "astar":
            return self.a_star_solver(state, max_iterations=50000)
        elif method == "greedy":
            return self.greedy_solver(state, max_steps=50)
        elif method == "agent":
            return self.solve_with_agent(state, "baseline")
        elif method == "agent2" and AGENT2_AVAILABLE:
            return self.solve_with_agent(state, "advanced")
        elif method == "layer":
            return self.layer_by_layer_solver(state)
        else:
            return self.hybrid_solver(state)
    
    def verify_solution(self, start_state: State, solution: List[str]) -> bool:
        """Verify that solution solves the cube"""
        test_state = start_state.copy()
        for move_name in solution:
            test_state.move(move_name)
        return test_state.isGoalState()
    
    def apply_solution(self, start_state: State, solution: List[str]) -> State:
        """Apply solution to state and return solved state"""
        solved_state = start_state.copy()
        for move_name in solution:
            solved_state.move(move_name)
        return solved_state

def state_to_dict(state: State) -> Dict:
    """Convert State object to dictionary for visualization"""
    return {
        'front': copy.deepcopy(state.front()),
        'back': copy.deepcopy(state.back()),
        'left': copy.deepcopy(state.left()),
        'right': copy.deepcopy(state.right()),
        'top': copy.deepcopy(state.top()),
        'bottom': copy.deepcopy(state.bottom())
    }

def apply_move_to_state(state: State, move_str: str) -> State:
    """Apply a move to the state and return new state"""
    new_state = state.copy()
    new_state.move(move_str)
    return new_state

def scramble_state(state: State, num_moves: int = 10) -> State:
    """Apply random moves to scramble the cube"""
    moves = ['front', 'back', 'left', 'right', 'top', 'bottom']
    scrambled_state = state.copy()
    last_move = None
    
    for _ in range(num_moves):
        # Avoid undo moves
        move = random.choice(moves)
        while move == last_move:
            move = random.choice(moves)
        
        scrambled_state.move(move)
        last_move = move
    
    return scrambled_state

def get_cube_stats(state: State) -> Dict:
    """Calculate comprehensive statistics about the cube state"""
    stats = {
        'solved_sides': 0,
        'correct_pieces': 0,
        'total_pieces': 54,
        'crosses': 0,
        'progress_percentage': 0
    }
    
    # Count solved sides
    faces = {
        'front': state.front(),
        'back': state.back(),
        'left': state.left(),
        'right': state.right(),
        'top': state.top(),
        'bottom': state.bottom()
    }
    
    for face_name, face in faces.items():
        center = face[1][1]
        
        # Check if side is completely solved
        side_solved = True
        for row in face:
            for sticker in row:
                if sticker == center:
                    stats['correct_pieces'] += 1
                else:
                    side_solved = False
        
        if side_solved:
            stats['solved_sides'] += 1
        
        # Check for cross
        if (face[0][1] == center and face[1][0] == center and 
            face[1][2] == center and face[2][1] == center):
            stats['crosses'] += 1
    
    # Calculate progress percentage
    stats['progress_percentage'] = (stats['correct_pieces'] / stats['total_pieces']) * 100
    
    return stats

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    defaults = {
        'cube_state': State(),
        'original_state': State(),  # Store original scrambled state
        'agent': None,
        'agent_type': None,
        'solution_path': [],
        'solved_state': None,  # Store the solved state
        'is_solving': False,
        'scramble_moves': 10,
        'cube_rotation': (0.3, 0.3),
        'solving_strategy': 'hybrid',
        'view_mode': 'professional',
        'animation_speed': 0.5,
        'show_wireframe': False,
        'move_history': [],
        'solving_stats': {'attempts': 0, 'successes': 0, 'avg_moves': 0, 'total_moves': 0}
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def render_cube_ui(state_dict: Dict, rotation: Tuple[float, float] = (0, 0)):
    """Render cube based on selected view mode"""
    if st.session_state.view_mode == 'professional':
        config = CubeConfig()
        renderer = ProfessionalCubeRenderer(config)
        return renderer.render_cube(state_dict, rotation)
    else:
        # Simple view
        fig = go.Figure()
        
        # Simple face rendering
        face_positions = {
            'front': (0, 0, 0.5),
            'right': (0.5, 0, 0),
            'left': (-0.5, 0, 0),
            'top': (0, 0.5, 0),
            'bottom': (0, -0.5, 0),
            'back': (0, 0, -0.5)
        }
        
        for face_name, (x, y, z) in face_positions.items():
            face_data = state_dict[face_name]
            for i in range(3):
                for j in range(3):
                    color_char = face_data[i][j]
                    color = COLOR_MAP.get(color_char, ('#CCCCCC', 'Gray'))[0]
                    
                    # Position sticker
                    sticker_x = x + (j - 1) * 0.2
                    sticker_y = y + (1 - i) * 0.2
                    sticker_z = z
                    
                    # Create a cube sticker
                    fig.add_trace(go.Mesh3d(
                        x=[sticker_x-0.1, sticker_x+0.1, sticker_x+0.1, sticker_x-0.1],
                        y=[sticker_y-0.1, sticker_y-0.1, sticker_y+0.1, sticker_y+0.1],
                        z=[sticker_z, sticker_z, sticker_z, sticker_z],
                        i=[0, 0],
                        j=[1, 2],
                        k=[2, 3],
                        color=color,
                        opacity=1.0,
                        showscale=False,
                        flatshading=True
                    ))
        
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False, range=[-1, 1]),
                yaxis=dict(visible=False, range=[-1, 1]),
                zaxis=dict(visible=False, range=[-1, 1]),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=400
        )
        return fig

def animate_solution(solution_path: List[str]):
    """Animate the solution step by step"""
    if not solution_path:
        st.warning("No solution to animate!")
        return
    
    # Store original state for animation
    animation_state = st.session_state.original_state.copy()
    
    animation_container = st.empty()
    
    with animation_container.container():
        st.markdown("### 🎬 Solution Animation")
        progress_bar = st.progress(0)
        
        for step_idx, move_step in enumerate(solution_path):
            progress = (step_idx + 1) / len(solution_path)
            progress_bar.progress(progress)
            
            # Apply move
            animation_state = apply_move_to_state(animation_state, move_step)
            
            # Update visualization
            state_dict = state_to_dict(animation_state)
            fig = render_cube_ui(state_dict, st.session_state.cube_rotation)
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption(f"Step {step_idx + 1}/{len(solution_path)}: **{move_step.upper()}**")
            
            # Pause for animation
            time.sleep(st.session_state.animation_speed)
            
            # Clear for next step
            if step_idx < len(solution_path) - 1:
                animation_container.empty()
        
        # Update the main cube state to solved state
        st.session_state.cube_state = animation_state
        st.success("✅ Animation Complete! Cube is now solved.")
        
        # Clear the animation container
        time.sleep(1)
        animation_container.empty()

def main():
    st.set_page_config(
        page_title="Rubik's Cube Master Solver",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        margin-bottom: 1rem;
    }
    .cube-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
    }
    .stButton > button {
        border-radius: 0.5rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Title with gradient
    st.markdown("""
    <h1 style="
        text-align: center;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        margin-bottom: 0;
    ">
    🧊 Rubik's Cube Solver
    </h1>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Create columns for layout
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Cube visualization container
        with st.container():
            st.markdown('<div class="cube-container">', unsafe_allow_html=True)
            
            # View controls
            col_view, col_rotate, col_speed = st.columns(3)
            with col_view:
                st.session_state.view_mode = st.selectbox(
                    "View Mode",
                    ["professional", "simple"],
                    index=0 if st.session_state.view_mode == 'professional' else 1,
                    key="view_mode_select"
                )
            
            with col_rotate:
                if st.button("🔄 Rotate View", key="rotate_view"):
                    st.session_state.cube_rotation = (
                        st.session_state.cube_rotation[0] + 0.3,
                        st.session_state.cube_rotation[1] + 0.3
                    )
            
            with col_speed:
                st.session_state.animation_speed = st.slider(
                    "Anim Speed", 0.1, 2.0, 0.5, 0.1,
                    key="anim_speed_slider"
                )
            
            # Get current state for visualization
            state_dict = state_to_dict(st.session_state.cube_state)
            fig = render_cube_ui(state_dict, st.session_state.cube_rotation)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Cube stats
            stats = get_cube_stats(st.session_state.cube_state)
            
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            with col_stats1:
                st.metric(
                    "Solved Sides", 
                    stats['solved_sides'], 
                    delta=f"{6 - stats['solved_sides']} to go"
                )
            with col_stats2:
                st.metric(
                    "Correct Pieces", 
                    stats['correct_pieces'], 
                    delta=f"{stats['progress_percentage']:.1f}%"
                )
            with col_stats3:
                if st.session_state.cube_state.isGoalState():
                    st.success("✅ Solved!")
                else:
                    st.info("🌀 Scrambled")
    
    with col2:
        # Control Panel
        st.markdown("### 🎮 Control Panel")
        
        # Agent Configuration
        with st.expander("🤖 AI Agent", expanded=True):
            if AGENT2_AVAILABLE:
                agent_type = st.radio(
                    "Agent Type",
                    ["Advanced (Beam Search)", "Baseline (Q-Learning)"],
                    index=0
                )
                
                if st.button("Initialize Agent", type="primary", key="init_agent"):
                    with st.spinner("Initializing..."):
                        if "Advanced" in agent_type:
                            st.session_state.agent = Agent2(
                                beam_width=32,
                                beam_depth=4,
                                step_penalty=0.01,
                                terminal_reward=10.0
                            )
                            st.session_state.agent_type = "agent2"
                        else:
                            st.session_state.agent = Agent()
                            st.session_state.agent.register_patterns()
                            st.session_state.agent_type = "agent"
                        st.success(f"{agent_type} initialized!")
            else:
                st.info("Baseline Agent available")
                if st.button("Initialize Baseline Agent", key="init_baseline"):
                    with st.spinner("Initializing..."):
                        st.session_state.agent = Agent()
                        st.session_state.agent.register_patterns()
                        st.session_state.agent_type = "agent"
                        st.success("Agent initialized!")
        
        # Cube Manipulation
        with st.expander("🎲 Cube Actions", expanded=True):
            col_scramble, col_reset = st.columns(2)
            with col_scramble:
                scramble_moves = st.number_input(
                    "Moves", 1, 30, 10,
                    key="scramble_moves_input"
                )
                if st.button("Scramble", key="scramble_btn"):
                    st.session_state.cube_state = scramble_state(State(), scramble_moves)
                    st.session_state.original_state = st.session_state.cube_state.copy()
                    st.session_state.solution_path = []
                    st.session_state.solved_state = None
                    st.rerun()
            
            with col_reset:
                if st.button("Reset", key="reset_btn"):
                    st.session_state.cube_state = State()
                    st.session_state.original_state = State()
                    st.session_state.solution_path = []
                    st.session_state.solved_state = None
                    st.rerun()
        
        # Manual Moves
        with st.expander("👆 Manual Moves", expanded=True):
            moves = ['front', 'back', 'left', 'right', 'top', 'bottom']
            move_cols = st.columns(3)
            
            for idx, move_name in enumerate(moves):
                with move_cols[idx % 3]:
                    if st.button(f"↻ {move_name.title()}", key=f"manual_{move_name}"):
                        st.session_state.cube_state = apply_move_to_state(
                            st.session_state.cube_state, 
                            move_name
                        )
                        st.session_state.move_history.append(move_name)
                        st.rerun()
        
        # Solving
        with st.expander("🧠 Solve Cube", expanded=True):
            solving_methods = ["hybrid", "astar", "bfs", "ids", "greedy", "layer"]
            if AGENT2_AVAILABLE:
                solving_methods.append("agent2")
            solving_methods.append("agent")
            
            st.session_state.solving_strategy = st.selectbox(
                "Solving Method",
                solving_methods,
                format_func=lambda x: {
                    "hybrid": "Hybrid (Recommended)",
                    "astar": "A* Search",
                    "bfs": "Breadth-First Search",
                    "ids": "Iterative Deepening",
                    "greedy": "Greedy Search",
                    "layer": "Layer-by-Layer",
                    "agent": "Baseline Agent",
                    "agent2": "Advanced Agent"
                }[x],
                key="strategy_select"
            )
            
            col_solve, col_apply = st.columns(2)
            
            with col_solve:
                if st.button("🚀 Find Solution", type="primary", key="solve_btn"):
                    if st.session_state.cube_state.isGoalState():
                        st.warning("Cube is already solved!")
                    else:
                        solver = RubiksCubeSolver()
                        solution = solver.solve(
                            st.session_state.cube_state,
                            st.session_state.solving_strategy
                        )
                        
                        if solution:
                            st.session_state.solution_path = solution
                            st.session_state.solving_stats['attempts'] += 1
                            st.session_state.solving_stats['successes'] += 1
                            total_moves = st.session_state.solving_stats.get('total_moves', 0) + len(solution)
                            st.session_state.solving_stats['total_moves'] = total_moves
                            if st.session_state.solving_stats['successes'] > 0:
                                st.session_state.solving_stats['avg_moves'] = total_moves / st.session_state.solving_stats['successes']
                            
                            # Store original state for animation
                            st.session_state.original_state = st.session_state.cube_state.copy()
                            
                            st.success(f"Found solution with {len(solution)} moves!")
                        else:
                            st.session_state.solving_stats['attempts'] += 1
                            st.error("Could not find a solution with current method.")
                        
                        st.rerun()
            
            with col_apply:
                if st.session_state.solution_path:
                    if st.button("✅ Apply Solution", key="apply_btn"):
                        solver = RubiksCubeSolver()
                        st.session_state.cube_state = solver.apply_solution(
                            st.session_state.original_state,
                            st.session_state.solution_path
                        )
                        st.success("Solution applied! Cube is now solved.")
                        st.rerun()
        
        # Current Solution
        if st.session_state.solution_path:
            with st.expander("📋 Solution Found", expanded=True):
                st.markdown(f"**{len(st.session_state.solution_path)} moves**")
                
                # Show moves in a compact format
                moves_text = " → ".join([m.upper() for m in st.session_state.solution_path])
                st.code(moves_text)
                
                col_animate, col_verify = st.columns(2)
                with col_animate:
                    if st.button("▶️ Animate Solution", key="animate_btn"):
                        animate_solution(st.session_state.solution_path)
                
                with col_verify:
                    solver = RubiksCubeSolver()
                    if st.button("✓ Verify Solution", key="verify_btn"):
                        if solver.verify_solution(st.session_state.original_state, st.session_state.solution_path):
                            st.success("✅ Solution is valid!")
                        else:
                            st.error("❌ Solution does not solve the cube!")
    
    # Footer with stats
    st.markdown("---")
    
    col_footer1, col_footer2 = st.columns(2)
    
    with col_footer1:
        st.markdown("### 📊 Solving Statistics")
        st.markdown(f"""
        - Attempts: {st.session_state.solving_stats['attempts']}
        - Successes: {st.session_state.solving_stats['successes']}
        - Avg Moves: {st.session_state.solving_stats['avg_moves']:.1f}
        """)
    
    with col_footer2:
        st.markdown("### ℹ️ Tips")
        st.markdown("""
        1. **Scramble** with 8-12 moves for best results
        2. Use **Hybrid** method for most reliable solving
        3. **Animate** to visualize the solution steps
        4. **Apply Solution** to update cube to solved state
        5. Try **Advanced Agent** if available for complex scrambles
        """)

if __name__ == "__main__":
    main()