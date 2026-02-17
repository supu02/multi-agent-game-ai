#!/usr/bin/env python3
"""
VAMPIRES VS WEREWOLVES - TOURNAMENT AI
======================================
Strategy: AGGRESSIVE HUMAN HUNTING + SMART COMBAT
- Race to humans FAST
- Convert everything we can
- Kill enemies when we're stronger
- Flee when they're stronger
- NO unnecessary splitting - stay together, stay strong!
"""

import sys
import time
import math
from collections import defaultdict
from argparse import ArgumentParser


# ============================================================
# CONFIGURATION
# ============================================================

TIME_BUDGET = 1.8  # Safe margin under 2s
MAX_DEPTH = 50     # Iterative deepening handles this
MAX_ACTIONS = 35   # Prune for speed

# ============================================================
# GLOBAL STATE
# ============================================================

GAME_STATE = {}

# ============================================================
# SERVER MESSAGE PARSING
# ============================================================

def UPDATE_GAME_STATE(message):
    if message is None:
        return

    typ = message[0]

    if typ == "set":
        n, m = message[1]
        GAME_STATE.clear()
        GAME_STATE.update({
            "rows": n,
            "cols": m,
            "cells": {},
            "hme": None,
            "my_species": None,
        })

    elif typ == "hme":
        GAME_STATE["hme"] = tuple(message[1])

    elif typ in ("map", "upd"):
        cells = GAME_STATE.setdefault("cells", {})
        for (x, y, H, V, W) in message[1]:
            if H == 0 and V == 0 and W == 0:
                cells.pop((x, y), None)
            else:
                cells[(x, y)] = (H, V, W)

        # Detect our species
        if GAME_STATE.get("my_species") is None and GAME_STATE.get("hme"):
            hx, hy = GAME_STATE["hme"]
            H, V, W = cells.get((hx, hy), (0, 0, 0))
            if V > 0:
                GAME_STATE["my_species"] = "V"
            elif W > 0:
                GAME_STATE["my_species"] = "W"


# ============================================================
# UTILITIES
# ============================================================

def dist(x1, y1, x2, y2):
    """Chebyshev distance - how many moves to reach target"""
    return max(abs(x2 - x1), abs(y2 - y1))


def in_bounds(x, y, rows, cols):
    return 0 <= x < cols and 0 <= y < rows


def get_neighbors(x, y, rows, cols):
    """All 8 directions"""
    result = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if in_bounds(nx, ny, rows, cols):
                result.append((nx, ny))
    return result


def step_toward(x, y, tx, ty):
    """One step toward target"""
    dx = 0 if tx == x else (1 if tx > x else -1)
    dy = 0 if ty == y else (1 if ty > y else -1)
    return x + dx, y + dy


def step_away(x, y, fx, fy):
    """One step away from threat"""
    dx = 0 if fx == x else (-1 if fx > x else 1)
    dy = 0 if fy == y else (-1 if fy > y else 1)
    return x + dx, y + dy


# ============================================================
# STATE EXTRACTION
# ============================================================

def get_my_groups(state, me):
    """Returns [(x, y, count), ...] sorted by count desc"""
    groups = []
    for (x, y), (H, V, W) in state.get("cells", {}).items():
        count = V if me == "V" else W
        if count > 0:
            groups.append((x, y, count))
    groups.sort(key=lambda g: g[2], reverse=True)
    return groups


def get_enemy_groups(state, me):
    """Returns [(x, y, count), ...]"""
    groups = []
    for (x, y), (H, V, W) in state.get("cells", {}).items():
        count = W if me == "V" else V
        if count > 0:
            groups.append((x, y, count))
    return groups


def get_humans(state):
    """Returns [(x, y, count), ...]"""
    humans = []
    for (x, y), (H, V, W) in state.get("cells", {}).items():
        if H > 0:
            humans.append((x, y, H))
    return humans


def total_units(state, me):
    return sum(g[2] for g in get_my_groups(state, me))


def total_enemy(state, me):
    return sum(g[2] for g in get_enemy_groups(state, me))


# ============================================================
# COMBAT MATH
# ============================================================

def win_prob(attacker, defender):
    """Probability attacker wins"""
    if attacker <= 0:
        return 0.0
    if defender <= 0:
        return 1.0
    if attacker == defender:
        return 0.5
    if attacker < defender:
        return attacker / (2.0 * defender)
    return min(1.0, attacker / defender - 0.5)


def can_convert(my_count, human_count):
    """Guaranteed conversion?"""
    return my_count >= human_count


def can_kill(my_count, enemy_count):
    """Guaranteed kill? (need 1.5x)"""
    return my_count >= 1.5 * enemy_count


def will_die(my_count, enemy_count):
    """Will enemy guaranteed kill us?"""
    return enemy_count >= 1.5 * my_count


def expected_vs_humans(attackers, humans):
    """Expected outcome attacking humans"""
    if attackers >= humans:
        return attackers + humans, 0
    P = win_prob(attackers, humans)
    return (attackers + humans) * P, humans * (1 - P)


def expected_vs_enemy(attackers, defenders):
    """Expected outcome attacking enemy"""
    if attackers >= 1.5 * defenders:
        return attackers, 0
    if defenders >= 1.5 * attackers:
        return 0, defenders
    P = win_prob(attackers, defenders)
    return attackers * P, defenders * (1 - P)


# ============================================================
# EVALUATION - SIMPLE BUT EFFECTIVE
# ============================================================

def evaluate(state, me):
    """
    Evaluate position. Higher = better for us.
    Focus on:
    1. Material (units we have vs enemy)
    2. Human potential (humans we can reach/convert)
    3. Safety (not being in danger)
    """
    my_groups = get_my_groups(state, me)
    enemy_groups = get_enemy_groups(state, me)
    humans = get_humans(state)
    
    my_total = sum(g[2] for g in my_groups)
    enemy_total = sum(g[2] for g in enemy_groups)
    human_total = sum(h[2] for h in humans)
    
    # Terminal states
    if my_total == 0:
        return -10000
    if enemy_total == 0 and human_total == 0:
        return 10000
    if enemy_total == 0:
        return 5000 + my_total  # Just need to clean up humans
    
    score = 0.0
    
    # === MATERIAL (most important!) ===
    score += 100 * (my_total - enemy_total)
    
    # Bonus for having overwhelming force
    if enemy_total > 0:
        ratio = my_total / enemy_total
        if ratio >= 1.5:
            score += 200  # We can kill them!
        elif ratio <= 0.67:
            score -= 200  # They can kill us!
    
    # === HUMAN HUNTING ===
    my_max = max((g[2] for g in my_groups), default=0)
    enemy_max = max((g[2] for g in enemy_groups), default=0)
    
    for (hx, hy, hcount) in humans:
        # Distance from our closest group
        my_dist = min((dist(gx, gy, hx, hy) for gx, gy, gc in my_groups), default=99)
        enemy_dist = min((dist(gx, gy, hx, hy) for gx, gy, gc in enemy_groups), default=99)
        
        # Can we convert them?
        we_can = my_max >= hcount
        they_can = enemy_max >= hcount
        
        if we_can:
            if my_dist < enemy_dist:
                # We'll get there first!
                score += 50 * hcount / (my_dist + 1)
            elif my_dist == enemy_dist:
                score += 20 * hcount / (my_dist + 1)
            else:
                score += 5 * hcount / (my_dist + 1)
        
        # Penalty if enemy will get them
        if they_can and enemy_dist < my_dist:
            score -= 30 * hcount / (enemy_dist + 1)
    
    # === PROXIMITY TO CONVERTIBLE HUMANS ===
    for (gx, gy, gc) in my_groups:
        for (hx, hy, hc) in humans:
            if gc >= hc:  # We can convert
                d = dist(gx, gy, hx, hy)
                score += 10 * hc / (d + 1)
    
    # === DANGER AVOIDANCE ===
    for (gx, gy, gc) in my_groups:
        for (ex, ey, ec) in enemy_groups:
            d = dist(gx, gy, ex, ey)
            if will_die(gc, ec):
                # Penalty for being close to stronger enemy
                if d <= 1:
                    score -= 500
                elif d <= 2:
                    score -= 200
                elif d <= 3:
                    score -= 50
            elif can_kill(gc, ec):
                # Bonus for being close to weaker enemy
                if d <= 2:
                    score += 80 * ec
    
    # === DANGER: Adjacent to big humans we can't convert ===
    for (gx, gy, gc) in my_groups:
        for (hx, hy, hc) in humans:
            if dist(gx, gy, hx, hy) == 1 and hc > gc:
                score -= 100 * (hc - gc)
    
    return score


# ============================================================
# SMART SPLITTING LOGIC
# ============================================================

def find_split_opportunity(gx, gy, gc, humans, enemy_groups, rows, cols, me):
    """
    Check if splitting this group makes sense.
    Returns split action [(x,y,n1,nx1,ny1), (x,y,n2,nx2,ny2)] or None.
    
    Split ONLY when:
    1. Multiple humans in DIFFERENT directions
    2. Each split can safely convert its target
    3. Enemy is far enough (safe to split)
    4. Worth it (total gain > going one by one)
    """
    
    # Need enough units to split meaningfully
    if gc < 4:
        return None
    
    # Check enemy distance - don't split if enemy is close!
    min_enemy_dist = 999
    max_enemy_size = 0
    for (ex, ey, ec) in enemy_groups:
        d = dist(gx, gy, ex, ey)
        if d < min_enemy_dist:
            min_enemy_dist = d
            max_enemy_size = ec
    
    # If enemy is close and dangerous, don't split
    if min_enemy_dist <= 4 and max_enemy_size >= gc * 0.5:
        return None
    
    # Find convertible humans in different directions
    convertible = []
    for (hx, hy, hc) in humans:
        d = dist(gx, gy, hx, hy)
        if d >= 1 and d <= 6:  # Reasonable range
            # Direction from us to human
            dx = 0 if hx == gx else (1 if hx > gx else -1)
            dy = 0 if hy == gy else (1 if hy > gy else -1)
            convertible.append((hx, hy, hc, d, dx, dy))
    
    if len(convertible) < 2:
        return None
    
    # Sort by distance (closest first)
    convertible.sort(key=lambda x: x[3])
    
    # Try to find two humans in DIFFERENT directions
    best_split = None
    best_gain = 0
    
    for i in range(len(convertible)):
        for j in range(i + 1, len(convertible)):
            h1 = convertible[i]  # (hx, hy, hc, d, dx, dy)
            h2 = convertible[j]
            
            hx1, hy1, hc1, d1, dx1, dy1 = h1
            hx2, hy2, hc2, d2, dx2, dy2 = h2
            
            # Must be in different directions (not same path)
            if (dx1, dy1) == (dx2, dy2):
                continue
            
            # Calculate split sizes needed
            need1 = hc1  # Need at least this many to convert
            need2 = hc2
            
            # Can we afford both splits?
            if need1 + need2 > gc:
                continue
            
            # Allocate units (minimum needed + fair share of remainder)
            remainder = gc - need1 - need2
            split1 = need1 + remainder // 2
            split2 = need2 + (remainder - remainder // 2)
            
            # Double-check both splits can convert
            if split1 < hc1 or split2 < hc2:
                continue
            
            # Calculate gain: getting both vs getting one then other
            # Split: d1 + d2 turns to get both (parallel)
            # No split: d1 + d2 turns anyway (sequential) BUT we're bigger for 2nd
            # Split is better when humans are in opposite-ish directions
            
            # Measure direction difference
            dir_diff = abs(dx1 - dx2) + abs(dy1 - dy2)
            
            if dir_diff >= 2:  # Significantly different directions
                # Worth splitting!
                total_gain = hc1 + hc2
                
                if total_gain > best_gain:
                    best_gain = total_gain
                    
                    # Calculate actual move targets (one step toward each)
                    nx1, ny1 = gx + dx1, gy + dy1
                    nx2, ny2 = gx + dx2, gy + dy2
                    
                    # Bounds check
                    if (0 <= nx1 < cols and 0 <= ny1 < rows and
                        0 <= nx2 < cols and 0 <= ny2 < rows):
                        best_split = [
                            (gx, gy, split1, nx1, ny1),
                            (gx, gy, split2, nx2, ny2)
                        ]
    
    return best_split


# ============================================================
# MOVE GENERATION - FOCUSED & FAST + SMART SPLITTING
# ============================================================

def generate_moves(state, me):
    """
    Generate sensible moves for all our groups.
    Each action = list of (x, y, count, nx, ny) moves.
    Includes SMART splitting when beneficial!
    """
    rows, cols = state["rows"], state["cols"]
    cells = state.get("cells", {})
    
    my_groups = get_my_groups(state, me)
    enemy_groups = get_enemy_groups(state, me)
    humans = get_humans(state)
    
    if not my_groups:
        return [[]]
    
    all_actions = []
    
    # === CHECK FOR SMART SPLITS (only for largest group) ===
    if my_groups and humans:
        gx, gy, gc = my_groups[0]  # Largest group
        split_action = find_split_opportunity(gx, gy, gc, humans, enemy_groups, rows, cols, me)
        if split_action:
            all_actions.append(split_action)
    
    # === REGULAR MOVES ===
    # For each group, find best moves
    group_best_moves = []
    
    for (gx, gy, gc) in my_groups[:3]:  # Top 3 groups max
        moves_for_group = []
        neighbors = get_neighbors(gx, gy, rows, cols)
        
        # Find targets
        best_human = None
        best_human_dist = 999
        for (hx, hy, hc) in humans:
            if gc >= hc:  # Can convert
                d = dist(gx, gy, hx, hy)
                if d < best_human_dist:
                    best_human_dist = d
                    best_human = (hx, hy, hc)
        
        # If no easy human, find ANY human (we might grow)
        if best_human is None and humans:
            for (hx, hy, hc) in humans:
                d = dist(gx, gy, hx, hy)
                if best_human is None or d < best_human_dist:
                    best_human_dist = d
                    best_human = (hx, hy, hc)
        
        best_kill = None
        for (ex, ey, ec) in enemy_groups:
            if can_kill(gc, ec):
                d = dist(gx, gy, ex, ey)
                if best_kill is None or d < dist(gx, gy, best_kill[0], best_kill[1]):
                    best_kill = (ex, ey, ec)
        
        threat = None
        for (ex, ey, ec) in enemy_groups:
            if will_die(gc, ec) and dist(gx, gy, ex, ey) <= 3:
                threat = (ex, ey, ec)
                break
        
        for (nx, ny) in neighbors:
            H, V, W = cells.get((nx, ny), (0, 0, 0))
            enemy_there = W if me == "V" else V
            
            # Don't walk into certain death
            if enemy_there > 0 and will_die(gc, enemy_there):
                continue
            
            # Don't step on humans we can't convert (risky)
            if H > 0 and H > gc and enemy_there == 0:
                continue
            
            # Score this move
            score = 0
            
            # Moving toward convertible humans = GREAT
            if best_human:
                hx, hy, hc = best_human
                old_d = dist(gx, gy, hx, hy)
                new_d = dist(nx, ny, hx, hy)
                if gc >= hc:  # Can convert
                    if new_d < old_d:
                        score += 100 + hc * 10
                    if (nx, ny) == (hx, hy):
                        score += 500 + hc * 20  # Actually converting!
                else:  # Can't convert yet, but move toward
                    if new_d < old_d:
                        score += 20
            
            # Moving toward killable enemy = GOOD
            if best_kill:
                ex, ey, ec = best_kill
                old_d = dist(gx, gy, ex, ey)
                new_d = dist(nx, ny, ex, ey)
                if new_d < old_d:
                    score += 50 + ec * 5
                if (nx, ny) == (ex, ey):
                    score += 300 + ec * 10  # Actually killing!
            
            # Fleeing from threat = NECESSARY
            if threat:
                tx, ty, tc = threat
                old_d = dist(gx, gy, tx, ty)
                new_d = dist(nx, ny, tx, ty)
                if new_d > old_d:
                    score += 200  # Getting away!
                elif new_d < old_d:
                    score -= 300  # Getting closer to death!
            
            # Moving to any human we can get
            if H > 0 and gc >= H and enemy_there == 0:
                score += 400 + H * 20
            
            moves_for_group.append((score, (gx, gy, gc, nx, ny)))
        
        # Sort by score
        moves_for_group.sort(reverse=True, key=lambda x: x[0])
        group_best_moves.append(moves_for_group[:5])  # Top 5 per group
    
    # === BUILD ACTIONS ===
    
    # Single group moves
    for group_moves in group_best_moves:
        for score, move in group_moves:
            all_actions.append([move])
    
    # Two groups moving together (if we have multiple groups already)
    if len(group_best_moves) >= 2:
        for s1, m1 in group_best_moves[0][:3]:
            for s2, m2 in group_best_moves[1][:3]:
                # Different sources
                if (m1[0], m1[1]) != (m2[0], m2[1]):
                    # Rule 5: source can't be target
                    src1, src2 = (m1[0], m1[1]), (m2[0], m2[1])
                    tgt1, tgt2 = (m1[3], m1[4]), (m2[3], m2[4])
                    if src1 != tgt2 and src2 != tgt1:
                        all_actions.append([m1, m2])
    
    if not all_actions:
        # Emergency: at least move somewhere
        if my_groups:
            gx, gy, gc = my_groups[0]
            neighbors = get_neighbors(gx, gy, rows, cols)
            if neighbors:
                nx, ny = neighbors[0]
                all_actions.append([(gx, gy, gc, nx, ny)])
    
    if not all_actions:
        all_actions = [[]]
    
    return all_actions[:MAX_ACTIONS]


# ============================================================
# APPLY ACTION
# ============================================================

def apply_action(state, action, player):
    """Apply action, return new state"""
    new_state = {
        "rows": state["rows"],
        "cols": state["cols"],
        "hme": state.get("hme"),
        "my_species": state.get("my_species"),
        "cells": dict(state["cells"]),
    }
    cells = new_state["cells"]
    
    if not action:
        return new_state
    
    # Remove from sources
    for (x, y, count, nx, ny) in action:
        H, V, W = cells.get((x, y), (0, 0, 0))
        if player == "V":
            V = max(0, V - count)
        else:
            W = max(0, W - count)
        if H == 0 and V == 0 and W == 0:
            cells.pop((x, y), None)
        else:
            cells[(x, y)] = (H, V, W)
    
    # Add to destinations (handle combat)
    arrivals = defaultdict(int)
    for (x, y, count, nx, ny) in action:
        arrivals[(nx, ny)] += count
    
    for (nx, ny), arriving in arrivals.items():
        H, V, W = cells.get((nx, ny), (0, 0, 0))
        my_there = V if player == "V" else W
        enemy_there = W if player == "V" else V
        
        total_my = my_there + arriving
        
        # VS ENEMY
        if enemy_there > 0 and H == 0:
            if total_my >= 1.5 * enemy_there:
                new_my, new_enemy = total_my, 0
            elif enemy_there >= 1.5 * total_my:
                new_my, new_enemy = 0, enemy_there
            else:
                exp_my, exp_en = expected_vs_enemy(total_my, enemy_there)
                new_my, new_enemy = int(round(exp_my)), int(round(exp_en))
            
            if player == "V":
                V, W = new_my, new_enemy
            else:
                W, V = new_my, new_enemy
            H = 0
        
        # VS HUMANS
        elif H > 0 and enemy_there == 0:
            if total_my >= H:
                new_my = total_my + H
                H = 0
            else:
                exp_my, exp_H = expected_vs_humans(total_my, H)
                new_my, H = int(round(exp_my)), int(round(exp_H))
            
            if player == "V":
                V = new_my
            else:
                W = new_my
        
        # EMPTY / OWN
        else:
            if player == "V":
                V = total_my
            else:
                W = total_my
        
        if H == 0 and V == 0 and W == 0:
            cells.pop((nx, ny), None)
        else:
            cells[(nx, ny)] = (H, V, W)
    
    return new_state


# ============================================================
# INSTANT WINNING MOVES
# ============================================================

def find_instant_win(state, me):
    """Find immediately winning moves - don't waste time searching!"""
    rows, cols = state["rows"], state["cols"]
    cells = state.get("cells", {})
    
    my_groups = get_my_groups(state, me)
    enemy_groups = get_enemy_groups(state, me)
    humans = get_humans(state)
    
    # Priority 1: Convert adjacent humans
    for (gx, gy, gc) in my_groups:
        for (hx, hy, hc) in humans:
            if dist(gx, gy, hx, hy) == 1 and gc >= hc:
                H, V, W = cells.get((hx, hy), (0, 0, 0))
                enemy = W if me == "V" else V
                if enemy == 0:
                    return [(gx, gy, gc, hx, hy)]
    
    # Priority 2: Kill adjacent weak enemy
    for (gx, gy, gc) in my_groups:
        for (ex, ey, ec) in enemy_groups:
            if dist(gx, gy, ex, ey) == 1 and can_kill(gc, ec):
                return [(gx, gy, gc, ex, ey)]
    
    # Priority 3: FLEE from adjacent deadly enemy
    for (gx, gy, gc) in my_groups:
        for (ex, ey, ec) in enemy_groups:
            d = dist(gx, gy, ex, ey)
            if d <= 2 and will_die(gc, ec):
                # Run away!
                nx, ny = step_away(gx, gy, ex, ey)
                if in_bounds(nx, ny, rows, cols):
                    # Make sure we're not running into another threat
                    H, V, W = cells.get((nx, ny), (0, 0, 0))
                    en = W if me == "V" else V
                    if not will_die(gc, en):
                        return [(gx, gy, gc, nx, ny)]
    
    return None


# ============================================================
# ALPHA-BETA SEARCH
# ============================================================

class Timeout(Exception):
    pass


def alphabeta(state, depth, alpha, beta, me, player, deadline):
    if time.time() > deadline:
        raise Timeout()
    
    if depth == 0:
        return evaluate(state, me), None
    
    my_total = total_units(state, me)
    enemy_total = total_enemy(state, me)
    
    if my_total == 0:
        return -10000, None
    if enemy_total == 0 and not get_humans(state):
        return 10000, None
    
    actions = generate_moves(state, player)
    if not actions or actions == [[]]:
        return evaluate(state, me), None
    
    # Sort actions by quick evaluation
    def action_score(a):
        if not a:
            return -9999
        s = 0
        cells = state.get("cells", {})
        for (x, y, c, nx, ny) in a:
            H, V, W = cells.get((nx, ny), (0, 0, 0))
            en = W if me == "V" else V
            if H > 0 and c >= H and en == 0:
                s += 1000 + H  # Conversion
            if en > 0 and can_kill(c, en):
                s += 800 + en  # Kill
        return s
    
    actions.sort(key=action_score, reverse=True)
    
    best_action = actions[0] if actions else None
    next_player = "W" if player == "V" else "V"
    
    if player == me:
        value = -float("inf")
        for action in actions:
            new_state = apply_action(state, action, player)
            score, _ = alphabeta(new_state, depth - 1, alpha, beta, me, next_player, deadline)
            if score > value:
                value = score
                best_action = action
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value, best_action
    else:
        value = float("inf")
        for action in actions:
            new_state = apply_action(state, action, player)
            score, _ = alphabeta(new_state, depth - 1, alpha, beta, me, next_player, deadline)
            if score < value:
                value = score
                best_action = action
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value, best_action


# ============================================================
# ITERATIVE DEEPENING
# ============================================================

def search(state, time_budget):
    me = state.get("my_species")
    if not me:
        return []
    
    # Check for instant wins first
    instant = find_instant_win(state, me)
    if instant:
        return instant
    
    deadline = time.time() + time_budget
    best_action = []
    
    # Fallback: generate at least one move
    actions = generate_moves(state, me)
    if actions and actions[0]:
        best_action = actions[0]
    
    # Iterative deepening
    for depth in range(1, MAX_DEPTH + 1):
        try:
            score, action = alphabeta(
                state, depth,
                -float("inf"), float("inf"),
                me, me, deadline
            )
            if action:
                best_action = action
            
            # Found a win?
            if score > 9000:
                break
            
            # Time check
            elapsed = time.time() - (deadline - time_budget)
            remaining = time_budget - elapsed
            if remaining < elapsed * 1.5:
                break
                
        except Timeout:
            break
    
    return best_action


# ============================================================
# MAIN INTERFACE
# ============================================================

def COMPUTE_NEXT_MOVE(state):
    me = state.get("my_species")
    if not me:
        return 0, []
    
    action = search(state, TIME_BUDGET)
    
    if not action:
        return 0, []
    
    return len(action), action


# ============================================================
# GAME LOOP
# ============================================================

def play_game(args):
    from client import ClientSocket
    
    client = ClientSocket(args.ip, args.port)
    client.send_nme("HunterBot")
    
    for _ in range(4):
        UPDATE_GAME_STATE(client.get_message())
    
    while True:
        msg = client.get_message()
        if msg is None:
            break
        
        typ = msg[0]
        
        if typ in ("set", "hum", "hme", "map", "upd"):
            UPDATE_GAME_STATE(msg)
        
        if typ == "upd":
            n, moves = COMPUTE_NEXT_MOVE(GAME_STATE)
            client.send_mov(n, moves)
        
        elif typ == "end":
            GAME_STATE.clear()
        
        elif typ == "bye":
            break


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = ArgumentParser()
    play_game(args=type('', (), {'ip': '127.0.0.1', 'port': 5555})())
