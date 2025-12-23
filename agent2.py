# agent2.py
# Stronger tabular agent with lookahead beam search + heuristic guidance.
# Works with your existing puzzle.State and UI.
#
# Highlights:
# - Heuristic: mismatch(s)/8 (cheap estimate of distance; lower is better)
# - Beam search: width W, depth D, iterative deepening 1..D
# - Score:  -α*h(next) + β*Q(s,a)  (tunable)
# - Action hygiene: no inverse / no immediate repeats in paths
# - Transposition pruning + beam dedup by hash
# - Random tie-breaks to escape plateaus
#
# You can still train Q tabularly via .train_episode(); beam works even with Q=0.

from __future__ import annotations
import random, copy, time
from collections import defaultdict
from typing import Dict, Tuple, List, Optional

from puzzle import State

ACTIONS = ["front","back","left","right","top","bottom"]
INVERSE = {"front":"back","back":"front","left":"right","right":"left","top":"bottom","bottom":"top"}

# ---------------- Utilities ----------------
def state_to_dict(s: State) -> Dict:
    return {
        "front":  [row[:] for row in s.front()],
        "back":   [row[:] for row in s.back()],
        "left":   [row[:] for row in s.left()],
        "right":  [row[:] for row in s.right()],
        "top":    [row[:] for row in s.top()],
        "bottom": [row[:] for row in s.bottom()],
    }

def mismatch_heuristic(sd: Dict) -> int:
    """# of stickers not matching center color. Range 0..54."""
    m = 0
    for f in ("front","back","left","right","top","bottom"):
        c = sd[f][1][1]
        for i in range(3):
            for j in range(3):
                if sd[f][i][j] != c:
                    m += 1
    return m

def epsilon_for_difficulty(diff: int, eps_max: float = 0.25) -> float:
    return min(eps_max, 0.02 + 0.23*(diff/54.0))

# ---------------- Simple Tabular Q ----------------
class QTable:
    def __init__(self, gamma=0.99, alpha=0.3):
        self.Q = defaultdict(float)
        self.gamma = gamma
        self.alpha = alpha

    def get(self, shash: int, a: str) -> float:
        return self.Q[(shash, a)]

    def set(self, shash: int, a: str, val: float):
        self.Q[(shash, a)] = val

    def best(self, shash: int) -> Tuple[str, float]:
        best_a, best_q = None, float("-inf")
        for a in ACTIONS:
            q = self.get(shash, a)
            if q > best_q:
                best_q, best_a = q, a
        return best_a, best_q

    def update(self, s: int, a: str, r: float, sp: int, done: bool):
        q = self.get(s, a)
        if done:
            target = r
        else:
            _, qn = self.best(sp)
            target = r + self.gamma * qn
        self.set(s, a, (1 - self.alpha) * q + self.alpha * target)

# ---------------- Agent2 (improved) ----------------
class Agent2:
    """
    Hybrid tabular agent with:
      - difficulty-aware epsilon for step policy (for quick runs),
      - strong beam search at inference for hard scrambles,
      - shaped rewards to allow optional tabular training.
    """
    def __init__(self,
                 gamma: float = 0.99,
                 alpha: float = 0.3,
                 step_penalty: float = 0.02,
                 terminal_reward: float = 1.0,
                 eps_max: float = 0.25,
                 beam_width: int = 32,
                 beam_depth: int = 3,
                 beam_alpha_h: float = 1.0,   # weight on heuristic (higher -> push toward progress)
                 beam_beta_q: float = 0.35,   # weight on Q bias (if you trained Q)
                 transpo_limit: int = 200000,
                 seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.Q = QTable(gamma=gamma, alpha=alpha)
        self.QV = self.Q.Q  # for UI compatibility

        self.step_penalty = step_penalty
        self.terminal_reward = terminal_reward
        self.eps_max = eps_max

        self.beam_width = max(0, int(beam_width))
        self.beam_depth = max(1, int(beam_depth))
        self.beam_alpha_h = float(beam_alpha_h)
        self.beam_beta_q = float(beam_beta_q)
        self.transpo_limit = int(transpo_limit)

        # book-keeping like your baseline
        self.last_action: Optional[str] = None
        self.second_last_action: Optional[str] = None

        # set by UI before solve
        self.curr_state: Optional[State] = None

    # ------------- Action hygiene -------------
    def legal_actions(self, last_a: Optional[str]) -> List[str]:
        acts = ACTIONS[:]
        if last_a:
            inv = INVERSE[last_a]
            if inv in acts: acts.remove(inv)
            if last_a in acts: acts.remove(last_a)
        return acts or ACTIONS

    # ------------- Greedy/ε action (used when beam disabled) -------------
    def select_action(self, s: State, epsilon: Optional[float] = None) -> str:
        sd = state_to_dict(s)
        diff = mismatch_heuristic(sd)
        eps = epsilon if epsilon is not None else epsilon_for_difficulty(diff, self.eps_max)

        if random.random() < eps:
            return random.choice(self.legal_actions(self.last_action))

        sh = s.__hash__()
        best_a, best_q = None, float("-inf")
        for a in self.legal_actions(self.last_action):
            q = self.Q.get(sh, a)
            if q > best_q:
                best_q, best_a = q, a
        if best_a is None:
            best_a = random.choice(ACTIONS)
        return best_a

    # ------------- Reward shaping (for training) -------------
    def shaped_reward(self, s: State, sp: State, done: bool) -> float:
        d0 = mismatch_heuristic(state_to_dict(s))
        d1 = mismatch_heuristic(state_to_dict(sp))
        delta = (d0 - d1) / 54.0
        r = delta - self.step_penalty
        if done:
            r += self.terminal_reward
        return r

    # ------------- Simple training loop (tabular) -------------
    def train_episode(self, epoch: int, max_steps: int = 300, depth: Optional[int] = None):
        # small curriculum
        if depth is None:
            if epoch <= 5:       depth = random.randint(1, 5)
            elif epoch <= 20:    depth = random.randint(1, min(25, 5 + (epoch-5)))
            else:                depth = random.randint(10, 30)

        s = State()
        last = None
        for _ in range(depth):
            a = random.choice(ACTIONS)
            if last and a == INVERSE[last]:
                continue
            s.move(a); last = a

        self.last_action = None
        self.second_last_action = None

        for _ in range(max_steps):
            a = self.select_action(s)
            sp = copy.deepcopy(s); sp.move(a)
            done = sp.isGoalState()
            r = self.shaped_reward(s, sp, done)
            self.Q.update(s.__hash__(), a, r, sp.__hash__(), done)
            s = sp
            self.second_last_action = self.last_action
            self.last_action = a
            if done:
                break

    # ------------- Public solve API -------------
    def evaluate(self, s: State, max_steps: int = 300) -> Tuple[bool, int, List[str]]:
        """
        Return (solved?, steps, path).
        If beam_width>0, we use iterative beam search depth 1..beam_depth each move.
        Otherwise, fallback to greedy ε-policy.
        """
        if s.isGoalState():
            return True, 0, []

        if self.beam_width <= 0:
            # greedy run
            steps, path = 0, []
            self.last_action = None
            self.second_last_action = None
            while steps < max_steps:
                steps += 1
                a = self.select_action(s)
                s.move(a)
                path.append(a)
                self.second_last_action = self.last_action
                self.last_action = a
                if s.isGoalState():
                    return True, steps, path
            return False, steps, path

        # beam-driven run
        solved, steps, path = self._solve_with_iterative_beam(s, max_steps=max_steps)
        return solved, steps, path

    # ------------- Beam machinery -------------
    def _score_edge(self, s: State, a: str) -> float:
        """Score for taking action a in state s (higher is better)."""
        sh = s.__hash__()
        q = self.Q.get(sh, a)
        sp = copy.deepcopy(s); sp.move(a)
        h = mismatch_heuristic(state_to_dict(sp)) / 8.0  # lower bound-ish
        # higher score when heuristic is small + Q is large
        return - self.beam_alpha_h * h + self.beam_beta_q * q

    def _expand(self, s: State, path: List[str], last_a: Optional[str]) -> List[Tuple[State, float, List[str]]]:
        out = []
        for a in self.legal_actions(last_a):
            sp = copy.deepcopy(s); sp.move(a)
            sc = self._score_edge(s, a)
            out.append((sp, sc, path + [a]))
        # small random jitter to break ties (stable but exploratory)
        random.shuffle(out)
        out.sort(key=lambda t: t[1], reverse=True)
        return out[:max(1, self.beam_width * 2)]  # over-generate; prune later

    def _solve_with_iterative_beam(self, s0: State, max_steps: int) -> Tuple[bool, int, List[str]]:
        Node = Tuple[State, float, List[str], Optional[str]]  # (state, score, path, last_a)
        steps = 0
        path_total: List[str] = []
        s = copy.deepcopy(s0)

        # Transposition table to avoid revisiting too much
        seen_counts: Dict[int, int] = {}
        seen_limit = self.transpo_limit

        while steps < max_steps:
            if s.isGoalState():
                return True, steps, path_total

            improved = False
            # try increasing lookahead depth from 1..D
            for depth in range(1, self.beam_depth + 1):
                ok, used, subpath = self._beam_search_once(s, depth, seen_counts, seen_limit)
                steps += used
                if ok and subpath:
                    # commit only first move; keep rest for next loop
                    a = subpath[0]
                    s.move(a)
                    path_total.append(a)
                    self.second_last_action = self.last_action
                    self.last_action = a
                    improved = True
                    break  # restart from new root

            if not improved:
                # fallback: one greedy step
                a = self.select_action(s)
                s.move(a)
                path_total.append(a)
                self.second_last_action = self.last_action
                self.last_action = a
                steps += 1

            if s.isGoalState():
                return True, steps, path_total

        return False, steps, path_total

    def _beam_search_once(self, s_root: State, depth: int,
                          seen: Dict[int,int], seen_limit: int) -> Tuple[bool, int, List[str]]:
        """
        One beam expansion from root up to 'depth'. Returns (found_better?, steps_used, path_from_root).
        We treat "better" as: we discovered a state with strictly lower heuristic h than root's h,
        or solved.
        """
        steps_used = 0
        h_root = mismatch_heuristic(state_to_dict(s_root)) / 8.0
        if h_root == 0:
            return True, steps_used, []

        # beam frontier: list of nodes
        frontier: List[Tuple[State, float, List[str], Optional[str]]] = [(copy.deepcopy(s_root), 0.0, [], None)]

        for d in range(1, depth + 1):
            cand: List[Tuple[State, float, List[str], Optional[str]]] = []
            for (s, score, path, last_a) in frontier:
                # expand
                for (sp, sc, p2) in self._expand(s, path, last_a):
                    h_sp = mismatch_heuristic(state_to_dict(sp)) / 8.0
                    steps_used += 1

                    # hit solved or strictly improved heuristic — return that path
                    if h_sp == 0:
                        return True, steps_used, p2
                    if h_sp < h_root:
                        return True, steps_used, p2

                    # transposition control
                    if len(seen) < seen_limit:
                        seen_hash = sp.__hash__()
                        c = seen.get(seen_hash, 0)
                        if c >= 3:
                            continue
                        seen[seen_hash] = c + 1

                    # collect candidate with combined score (prefer good Q and low h)
                    total_score = score + (- self.beam_alpha_h * h_sp) + self.beam_beta_q * self.Q.get(s.__hash__(), p2[-1])
                    cand.append((sp, total_score, p2, p2[-1]))

            if not cand:
                break

            # de-duplicate by hash; keep the best scoring per hash
            by_hash: Dict[int, Tuple[State, float, List[str], Optional[str]]] = {}
            for node in cand:
                hsh = node[0].__hash__()
                if (hsh not in by_hash) or (node[1] > by_hash[hsh][1]):
                    by_hash[hsh] = node
            uniq = list(by_hash.values())

            # prune to beam width; small random shuffle to avoid ties lock
            random.shuffle(uniq)
            uniq.sort(key=lambda t: t[1], reverse=True)
            frontier = uniq[:self.beam_width]

        # no immediate improvement found
        return False, steps_used, []
