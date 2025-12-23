# Rubik’s Cube Solver — Streamlit + Search + Agents

Click the link to run live on Streamlit: https://rubik-cube-jne2ldq82akvf5cnrm3ukm.streamlit.app/

This repository hosts an interactive Rubik’s Cube solver with a **Streamlit** web UI and multiple solving strategies:
- Graph search (BFS / IDS / A* / Greedy)
- A heuristic “layer-by-layer” improver
- Agent-based solvers (**tabular Q** + an **optional** stronger beam-search agent)

The app is designed for **short scrambles** (the cube state-space is huge; these solvers are not a full Kociemba/CFOP production solver).

---

## Quick start (local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Deploy to Streamlit Community Cloud (get a shareable link)

1. Push this repo to GitHub.
2. On Streamlit Community Cloud → **New app**
3. Select your repo and set:
   - **Main file path**: `app.py`
4. Deploy → you’ll get a link like:
   `https://<your-app-name>.streamlit.app`

---

## Repository layout

```text
.
├── app.py            # Streamlit entrypoint (UI + renderer + solver orchestration)
├── puzzle.py         # Cube state (State), 180° move operators, helper metrics
├── Agent.py          # Baseline tabular Q-learning style agent (pattern-seeded)
├── agent2.py         # Optional advanced agent (beam search + heuristic + shaped reward)
├── requirements.txt  # streamlit, plotly, numpy
└── .streamlit/
    └── config.toml   # optional Streamlit config
```

> **Important (Linux case-sensitivity):** Streamlit Cloud runs on Linux.  
> Because `app.py` imports `from Agent import Agent`, the file must be named **exactly** `Agent.py` (capital A).

---

# File-by-file deep dive

## 1) `app.py` — Streamlit UI + 3D renderer + solver glue

`app.py` is the **main entrypoint** and contains three big parts:

### A) Imports and module wiring
- Imports `State` and the functional `move(s, action)` helper from `puzzle.py`.
- Imports `Agent` from `Agent.py`.
- Tries to import `Agent2` from `agent2.py` (if present, the UI enables it; otherwise it falls back gracefully).

### B) 3D cube visualization: `ProfessionalCubeRenderer`
**Class: `ProfessionalCubeRenderer`**
- **Purpose:** Turn a cube state into a **Plotly 3D** visualization.
- **Key methods:**
  - `render_cube(state_dict, rotation)`  
    Builds a `plotly.graph_objects.Figure` and draws the cube with shaded “cubies” and colored stickers.
  - `_create_piece(...)`  
    Internal helper that draws each visible piece surface.

**Data format used by the renderer**
- The renderer expects a **dictionary** with 6 faces: `front/back/left/right/top/bottom`,
  each face being a `3×3` grid of sticker letters (e.g., `"W", "R", ..."`).
- Conversion happens via:

**Function:** `state_to_dict(state: State) -> dict`  
Returns a deep-copied dict of the 6 faces for safe rendering.

### C) Solving algorithms: `RubiksCubeSolver`
**Class: `RubiksCubeSolver`** orchestrates every solving method.  
It defines:
- `self.moves`: `['front','back','left','right','top','bottom']`
- `self.inverse_moves`: inverse mapping used for pruning (avoid immediate backtracking)

#### C1) Heuristic: `get_heuristic(state) -> int`
- Computes a **mismatch count**: number of stickers not matching their face center.
- Adds an extra **corner penalty** based on “corner color uniqueness” to discourage inconsistent corner triples.
- Lower score means “closer to solved”.

This heuristic is used by:
- A*
- Greedy best-first
- Layer-by-layer improver
- Hybrid method ordering

#### C2) BFS: `bfs_solver(start_state, max_depth=20) -> List[str]`
- Standard breadth-first search:
  - queue holds `(state, path)`
  - a `visited` set stores `state.__hash__()` to avoid repeats
  - stops when the goal is found or the queue is exhausted
- Depth-limited via `max_depth` to prevent runaway search.

**Best for:** very small scrambles (few moves).  
**Tradeoff:** memory grows exponentially.

#### C3) IDS / DLS: `ids_solver` and `dls_solver`
- `ids_solver(start_state, max_depth=20)` runs iterative deepening:
  - tries depth=0..max_depth
  - calls `dls_solver` each time
- `dls_solver(state, depth, path, visited)` is a depth-limited DFS.

**Best for:** small memory footprint compared to BFS.  
**Tradeoff:** repeats work at each depth, still exponential time.

#### C4) Move ordering: `get_ordered_moves(state) -> List[str]`
- Computes the heuristic score for each candidate move.
- Sorts moves so the search tries “most promising” actions earlier.
- Used by A* and Greedy.

#### C5) A*: `a_star_solver(start_state, max_depth=25) -> List[str]`
- Priority queue (heap) of nodes ordered by:
  - `f = g + h`, where `g = path length`, `h = get_heuristic(state)`
- Uses `visited`/`cost_so_far` bookkeeping by hash to reduce revisits.
- Limited by `max_depth` for safety.

**Best for:** short scrambles, when heuristic is informative.  
**Tradeoff:** weak heuristic can still cause many expansions.

#### C6) Greedy best-first: `greedy_solver(start_state, max_steps=30) -> List[str]`
- Always picks the next move that **minimizes heuristic** immediately.
- Includes simple “anti-loop” logic via state hashes / move pruning.

**Best for:** easy states where heuristic gradient is smooth.  
**Tradeoff:** can get stuck in local minima; not optimal.

#### C7) Agent solving: `solve_with_agent(start_state, agent_type='baseline'|'agent2', ...)`
- **Baseline:** uses `Agent.Play()` logic from `Agent.py`.
- **Advanced:** uses `Agent2.evaluate(...)` from `agent2.py` if available.
- Returns move list + metadata (steps, solved flag, etc.).

#### C8) Layer-by-layer improver: `layer_by_layer_solver(start_state, max_steps=50)`
- Not a full CFOP/Kociemba algorithm.
- A heuristic “improve the score” routine that tries moves that reduce the heuristic score.
- Works as a simple, human-readable method.

#### C9) Hybrid: `hybrid_solver(start_state)`
- Attempts multiple strategies in a practical order (e.g., try faster methods first).
- Returns the first valid solution found.

#### C10) Verification + application
- `verify_solution(start_state, solution)`  
  Applies moves to a copy and checks if the result is solved.
- `apply_solution(start_state, solution)`  
  Returns the resulting state after applying the move sequence.

### D) Streamlit session state and UI helpers
**Function:** `initialize_session_state()`
- Stores persistent objects in `st.session_state`:
  - current cube state
  - move history
  - current solver choice
  - last solution path (for animation)
  - stats counters

**Function:** `scramble_state(state, num_moves)`
- Applies a random scramble sequence (using valid moves).

**Function:** `render_cube_ui(state_dict, rotation)`
- Calls the renderer and embeds the Plotly figure into Streamlit.

**Function:** `animate_solution(solution_path)`
- Steps through moves with a short delay to animate solving.

**Function:** `main()`
- The Streamlit page layout:
  - sidebar controls (scramble, solver selection, parameters)
  - center visualization
  - solve button + output metrics

---

## 2) `puzzle.py` — Cube environment (state + 180° moves + metrics)

`puzzle.py` defines the cube dynamics used by both search solvers and agents.

### A) Core data structure: `State`
**Class: `State(size=3, c=None)`**
- Represents a cube with 6 faces, each a `3×3` grid.
- Holds an action list:
  - `['front','back','left','right','top','bottom']`

**Key methods**
- `copy()`  
  Deep-copies the cube state safely.
- `__str__()` / `__hash__()`  
  Hashing is based on string serialization → used by visited sets in search.
- `front()/back()/left()/right()/top()/bottom()`  
  Face accessors used by the UI and heuristics.
- `rotate_side(side)`  
  Performs a **180° rotation** of a 3×3 face.
- `flip_cube()`  
  Internal helper used by some move implementations to reuse logic.
- `turn_front / turn_back / turn_left / turn_right / turn_top / turn_bottom`  
  Apply one of the 6 moves by:
  - rotating the target face (180°)
  - permuting edge strips on adjacent faces
- `move(action)`  
  Dispatches the move name to the corresponding `turn_*` method and refreshes cached side list.
- `isGoalState()`  
  Returns True when each face is uniform (all stickers match the face center).

> **Design note:** moves in this code are **half-turn (180°)** moves for each face, not the full 18-move cube action set.
> This reduces branching and makes search more feasible for short scrambles.

### B) Functional helpers (pure functions)
These are used heavily by the agents/search code:

- `move(s: State, action: str) -> State`  
  Returns a *new* state: copies `s`, applies the action, returns the result.

- `random_move(cube)` / `shuffle(cube, n)`  
  Apply random actions to generate scrambles.

- `one_move_state()` / `n_move_state(n)`  
  Convenience constructors that start from solved and apply `n` random moves.

### C) Progress metrics (used as heuristic features / reward shaping)
- `num_pieces_correct_side(state)`  
  Counts stickers matching their face center color.
- `num_solved_sides(state)`  
  Number of faces that are fully solved (all 9 stickers match center).
- `num_crosses(state)` / `num_xs(state)`  
  Simple pattern counters helpful for shaping “partial progress” rewards.

---

## 3) `Agent.py` — Baseline agent (pattern-seeded tabular Q)

`Agent.py` implements a lightweight agent that uses a **Q-value table** for state-action pairs.

### A) What it stores
Inside `Agent`:
- `self.QV`: dictionary of Q-values keyed by `(state_hash, action)`
- `self.R`: dictionary mapping `state_hash -> list of per-action rewards` (memoization)
- `self.visited`, `self.visit_count`: basic visitation tracking
- Pattern buffers:
  - `self.one_away, two_away, ... six_away`

### B) Pattern seeding: `register_patterns()`
Instead of learning from scratch across the full state space, it seeds Q-values by enumerating states that are
**1 to 6 moves away** from the goal:
- For a state one move away from goal: correct action gets +10, others -10
- Two away: ±6
- Three away: ±5
- Four away: ±4
- Five away: ±3
- Six away: ±1

This gives the agent “knowledge” near the solved state, which is why it performs best on short scrambles.

### C) Learning loop: `QLearn(discount=0.99, episodes=10, epsilon=0.9)`
- Runs an epsilon-greedy loop:
  - with probability `epsilon`, explore random actions
  - otherwise exploit best known Q-value
- Updates Q-values with a standard temporal-difference style update:
  - `Q(s,a) ← (1-α)Q(s,a) + α[r + γ max_a' Q(s',a')]`
- Uses `reward(state, action)` and `max_reward(state, action)` for scoring.

### D) Acting / rollout: `Play()`
- Starts from `self.start_state` and tries up to 20 steps.
- Chooses the best action by Q-value (with some randomness / anti-repeat logic).
- Stops if the cube becomes solved.

### E) Reward shaping: `reward(state, action)`
- Large terminal reward if goal reached.
- Small step penalty otherwise.
- Uses cube metrics from `puzzle.py`:
  - `num_solved_sides`
  - `num_pieces_correct_side`
- Penalizes regressions (when next state is worse than current state).

> **Why this is “baseline”:** tabular Q-learning does not scale to the real cube without large abstraction or function approximation.
> Here it is intentionally small and educational, and works best near solved states.

---

## 4) `agent2.py` — Advanced hybrid agent (beam search + heuristic + shaped reward)

`agent2.py` adds a stronger approach that mixes:
- a small Q-table bias (optional)
- a fast heuristic (mismatch count)
- **lookahead beam search** to choose actions more robustly

### A) Utility functions
- `state_to_dict(s: State) -> dict`  
  Converts `State` into face dict format (same structure used by `app.py`).
- `mismatch_heuristic(sd: dict) -> int`  
  Counts stickers not matching face center (0..54). Lower is better.
- `epsilon_for_difficulty(diff, eps_max)`  
  Makes exploration rate depend on difficulty.

### B) `QTable` helper class
A simple wrapper around a dict:
- `get(key, default)`
- `set(key, value)`
- `best(state_hash, actions)` → returns the best action/value known
- `update(...)` → TD-style update

### C) `Agent2` class
**Constructor parameters (important knobs)**
- `beam_width`: number of frontier nodes kept each depth
- `beam_depth`: how far to look ahead each decision
- `beam_alpha_h`: weight of heuristic progress
- `beam_beta_q`: weight of Q-table bias (if trained)
- `step_penalty`, `terminal_reward`: shaped reward

**Key public methods**
- `train_episode(start_state, max_steps)`  
  Runs one training episode with shaped rewards (optional).
- `evaluate(state, max_steps)`  
  Main inference method used by the UI:
  - iteratively selects moves
  - uses beam search lookahead
  - returns `(solved, steps, path)`

**Beam search internals**
- `_beam_search_once(state, last_action, second_last_action)`  
  Expands a search tree for a few steps:
  - keeps only top `beam_width` nodes by score
  - deduplicates states by hash (transposition pruning)
- `_score_edge(s, a, s_next)`  
  Scores candidate edges using a combination of:
  - heuristic improvement
  - Q bias (if present)
  - step penalty

**Why Agent2 tends to work better**
- Greedy one-step choice can get trapped.
- Beam search explores multiple promising trajectories and picks the best path prefix.
- Reward shaping helps it prefer “net progress” rather than oscillating.

---

# How the files connect (call graph overview)

1. UI in `app.py` holds the cube state (`puzzle.State`) in `st.session_state`.
2. When you click **Scramble**, `app.py` calls `scramble_state(...)` → uses `State.move(...)`.
3. When you click **Solve**:
   - Search-based methods call `RubiksCubeSolver.solve(...)` → BFS/IDS/A*/Greedy/etc.
   - Agent methods call `RubiksCubeSolver.solve_with_agent(...)`:
     - Baseline agent: `Agent.Play()`
     - Advanced agent: `Agent2.evaluate(...)`
4. Any solution path is animated in the UI using repeated `State.move(...)` calls.

---

# Practical tips / extension points

## Add full cube move set (clockwise, counterclockwise, 180)
Right now actions are only the 6 faces with a single fixed turn.  
To extend:
- represent actions as `(face, direction)` and implement `turn_front_cw`, `turn_front_ccw`, etc.
- update the solver move lists and inverse mapping accordingly

## Improve heuristics
- The mismatch heuristic is simple but weak.
- Big upgrades:
  - pattern databases (PDB)
  - learned value function
  - two-phase solver heuristics (Kociemba-style)

## Make it “real cube solver” quality
For real-world random scrambles, consider integrating:
- Kociemba 2-phase solver (external library)
- CFOP algorithm tables
- pruning + symmetry reduction

---

# Troubleshooting

### Import errors on Streamlit Cloud
- Ensure `Agent.py` uses a capital **A** (Linux is case-sensitive).
- Ensure all `.py` files are in the repo root (same folder as `app.py`).

### Solver fails on long scrambles
- This is expected for search + small tabular agents.
- Try fewer scramble moves, Hybrid, or the advanced agent.

---


