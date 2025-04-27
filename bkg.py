#!/usr/bin/env python3
import random
import datetime
import copy
import os
import time
import json
import math
import traceback
from dataclasses import dataclass

# --- Constants ---
MAX_DEPTH = 3             # Minimax search depth
NUM_DICE_SAMPLES = 14     # Number of dice samples for Minimax chance nodes

# --- Heuristic Weights Definition ---
@dataclass
class HeuristicWeights:
    """Stores weights for different heuristic components."""
    PIP_SCORE_FACTOR: float = 1.0
    OFF_SCORE_FACTOR: float = 10.0
    HIT_BONUS: float = 30.0
    BAR_PENALTY: float = -20.0
    POINT_BONUS: float = 2.0
    HOME_BOARD_POINT_BONUS: float = 3.0
    INNER_HOME_POINT_BONUS: float = 2.0
    ANCHOR_BONUS: float = 5.0
    PRIME_BASE_BONUS: float = 4.0
    DIRECT_SHOT_PENALTY_FACTOR: float = -1.5
    BLOT_PENALTY_REDUCTION_IF_OPP_ON_BAR: float = 0.5
    AGGRESSION_THRESHOLD: float = 10.0
    MIDGAME_HOME_PRISON_BONUS: float = 20.0
    FAR_BEHIND_BACK_CHECKER_PENALTY_FACTOR: float = 0.0
    TRAPPED_CHECKER_BONUS: float = 8.0

# Define weights for different game phases
OPENING_WEIGHTS = HeuristicWeights(
    PIP_SCORE_FACTOR=0.8, OFF_SCORE_FACTOR=5.0, HIT_BONUS=35.0,
    BAR_PENALTY=-25.0, POINT_BONUS=3.0, HOME_BOARD_POINT_BONUS=2.0,
    INNER_HOME_POINT_BONUS=1.0, ANCHOR_BONUS=8.0, PRIME_BASE_BONUS=5.0,
    DIRECT_SHOT_PENALTY_FACTOR=-1.0,
    BLOT_PENALTY_REDUCTION_IF_OPP_ON_BAR=0.6, AGGRESSION_THRESHOLD=12.0,
    MIDGAME_HOME_PRISON_BONUS=15.0,
    FAR_BEHIND_BACK_CHECKER_PENALTY_FACTOR=0.5, TRAPPED_CHECKER_BONUS=6.0
)
MIDGAME_WEIGHTS = HeuristicWeights(
    PIP_SCORE_FACTOR=1.2, OFF_SCORE_FACTOR=15.0, HIT_BONUS=40.0,
    BAR_PENALTY=-25.0, POINT_BONUS=3.0, HOME_BOARD_POINT_BONUS=5.0,
    INNER_HOME_POINT_BONUS=3.0, ANCHOR_BONUS=3.0, PRIME_BASE_BONUS=5.0,
    DIRECT_SHOT_PENALTY_FACTOR=-1.5,
    BLOT_PENALTY_REDUCTION_IF_OPP_ON_BAR=0.5, AGGRESSION_THRESHOLD=15.0,
    MIDGAME_HOME_PRISON_BONUS=20.0,
    FAR_BEHIND_BACK_CHECKER_PENALTY_FACTOR=0.7, TRAPPED_CHECKER_BONUS=8.0
)
ENDGAME_WEIGHTS = HeuristicWeights(
    PIP_SCORE_FACTOR=3.0, OFF_SCORE_FACTOR=30.0, HIT_BONUS=50.0,
    BAR_PENALTY=-50.0, POINT_BONUS=0.5, HOME_BOARD_POINT_BONUS=0.5,
    INNER_HOME_POINT_BONUS=0.2, ANCHOR_BONUS=1.0, PRIME_BASE_BONUS=1.0,
    DIRECT_SHOT_PENALTY_FACTOR=-2.5,
    BLOT_PENALTY_REDUCTION_IF_OPP_ON_BAR=0.2, AGGRESSION_THRESHOLD=5.0,
    MIDGAME_HOME_PRISON_BONUS=0.0,
    FAR_BEHIND_BACK_CHECKER_PENALTY_FACTOR=0.0, TRAPPED_CHECKER_BONUS=2.0
)
PHASE_WEIGHTS = {
    'OPENING': OPENING_WEIGHTS,
    'MIDGAME': MIDGAME_WEIGHTS,
    'ENDGAME': ENDGAME_WEIGHTS
}


# --- Backgammon Game Class ---
class BackgammonGame:
    """Manages game state, rules, evaluation for Human vs AI play."""
    OPENING_EXIT_TOTAL_PIP_THRESHOLD = 280

    def __init__(self, human_player=None):
        """Initializes the game board and state."""
        self.board = [
             2, 0, 0, 0, 0,-5, 0,-3, 0, 0, 0, 5,  # Pts 1-12 (idx 0-11)
            -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0,-2   # Pts 13-24 (idx 12-23)
        ]
        self.white_bar = 0
        self.black_bar = 0
        self.white_off = 0
        self.black_off = 0
        self.winner = None
        self.current_player = 'w'
        self.dice = []
        self.available_moves = []
        self.human_player = human_player
        self.ai_player = None
        if human_player == 'w':
            self.ai_player = 'b'
        elif human_player == 'b':
            self.ai_player = 'w'
        self.current_phase = self.determine_game_phase()
        self.white_last_turn_sequence = []
        self.black_last_turn_sequence = []

    def copy(self):
        """Creates a deep copy for AI simulation."""
        new_game = BackgammonGame(human_player=self.human_player)
        new_game.board = list(self.board)
        new_game.white_bar = self.white_bar
        new_game.black_bar = self.black_bar
        new_game.white_off = self.white_off
        new_game.black_off = self.black_off
        new_game.winner = self.winner
        new_game.current_player = self.current_player
        new_game.dice = list(self.dice)
        new_game.available_moves = list(self.available_moves)
        new_game.current_phase = self.current_phase
        new_game.white_last_turn_sequence = list(self.white_last_turn_sequence)
        new_game.black_last_turn_sequence = list(self.black_last_turn_sequence)
        return new_game

    def calculate_pip(self, player):
        """Calculates the pip count for a player."""
        pip = 0
        p_sign = 1 if player == 'w' else -1
        p_bar_count = self.white_bar if player == 'w' else self.black_bar

        for pos in range(1, 25):
            point_index = pos - 1
            count = self.board[point_index]
            if count * p_sign > 0:
                distance = (25 - pos) if player == 'w' else pos
                pip += distance * abs(count)

        pip += p_bar_count * 25
        return pip

    def is_game_over(self):
        """Checks if the game has ended."""
        if self.white_off >= 15:
            self.winner = 'w'
            return True
        if self.black_off >= 15:
            self.winner = 'b'
            return True
        return False

    def board_tuple(self):
        """Returns a hashable representation of the core game state."""
        return (
            tuple(self.board), self.white_bar, self.black_bar,
            self.white_off, self.black_off, self.current_player
        )

    def get_total_checker_count(self):
        """Counts all checkers for validation."""
        w_board = sum(c for c in self.board if c > 0)
        b_board = sum(abs(c) for c in self.board if c < 0)
        w_total = w_board + self.white_bar + self.white_off
        b_total = b_board + self.black_bar + self.black_off
        return w_total, b_total

    def parse_move(self, move_str, current_player_hint=None):
        """Parses 'src/dst' input string into move components."""
        try:
            move_str = move_str.lower().strip()
            src, dst = None, None
            if '/' not in move_str: return None, None, None
            src_str, dst_str = move_str.split('/', 1)
            player = current_player_hint

            if src_str == 'bar': src = 'bar'
            elif src_str.isdigit():
                val = int(src_str)
                if 1 <= val <= 24: src = val
                else: return None, None, None
            else: return None, None, None

            if dst_str == 'off': dst = 'off'
            elif dst_str.isdigit():
                val = int(dst_str)
                if 1 <= val <= 24: dst = val
                else: return None, None, None
            else: return None, None, None

            if src is None or dst is None: return None, None, None
            if src == 'bar' and dst == 'off': return None, None, None
            if isinstance(src, int) and isinstance(dst, int) and src == dst:
                return None, None, None
            return player, src, dst
        except Exception: return None, None, None

    # --- Rule Functions ---
    def _check_all_pieces_home(self, player, game_state):
        """Checks if all player's pieces are in their home board."""
        p_sign = 1 if player == 'w' else -1
        bar_count = game_state.white_bar if player == 'w' \
            else game_state.black_bar
        if bar_count > 0: return False

        outside_range = range(1, 19) if player == 'w' else range(7, 25)
        for pos in outside_range:
            point_index = pos - 1
            if game_state.board[point_index] * p_sign > 0: return False
        return True

    def _can_bear_off(self, player, checker_pos, die_value, game_state):
        """Checks if a specific checker can be legally borne off."""
        p_sign = 1 if player == 'w' else -1
        board = game_state.board
        home_start, home_end = (19, 24) if player == 'w' else (1, 6)

        if not (home_start <= checker_pos <= home_end): return False

        target_point_for_exact_off = 25 if player == 'w' else 0
        required_dist = abs(target_point_for_exact_off - checker_pos)

        if die_value == required_dist: return True

        is_overshoot = (player == 'w' and checker_pos + die_value > 24) or \
                       (player == 'b' and checker_pos - die_value < 1)

        if is_overshoot and die_value > required_dist:
             check_behind_range = range(home_start, checker_pos) if player == 'w' \
                 else range(checker_pos + 1, home_end + 1)
             for p in check_behind_range:
                 point_index = p - 1
                 if board[point_index] * p_sign > 0: return False
             return True
        return False

    def _get_die_for_move(self, src, dst, player):
        """Determines the nominal die value for a move (ignores overshoot)."""
        try:
            if isinstance(src, int) and isinstance(dst, int):
                diff = (dst - src) if player == 'w' else (src - dst)
                return diff if 1 <= diff <= 6 else None
            elif src == 'bar' and isinstance(dst, int):
                 entry_point = dst
                 die = entry_point if player == 'w' else (25 - entry_point)
                 return die if 1 <= die <= 6 else None
            elif isinstance(src, int) and dst == 'off':
                 if player == 'w' and not (19 <= src <= 24): return None
                 if player == 'b' and not (1 <= src <= 6): return None
                 needed = (25 - src) if player == 'w' else src
                 return needed if 1 <= needed <= 6 else None
            else: return None
        except Exception: return None

    def _get_single_moves_for_die(self, player, die_value, current_state):
        """Finds all possible single (src, dst) moves for ONE die value."""
        moves = []
        p_sign = 1 if player == 'w' else -1
        p_bar = current_state.white_bar if player == 'w' \
            else current_state.black_bar
        board = current_state.board

        if p_bar > 0:
            entry_point = die_value if player == 'w' else (25 - die_value)
            if 1 <= entry_point <= 24:
                entry_idx = entry_point - 1
                dest_count = board[entry_idx]
                is_allowed_entry = (dest_count * p_sign >= 0) or \
                                   (abs(dest_count) == 1)
                if is_allowed_entry:
                    moves.append(('bar', entry_point))
            return moves

        all_checkers_home = self._check_all_pieces_home(player, current_state)
        for pos in range(1, 25):
            pos_idx = pos - 1
            checker_count = board[pos_idx]

            if checker_count * p_sign > 0:
                dest_point = (pos + die_value) if player == 'w' \
                    else (pos - die_value)
                if 1 <= dest_point <= 24:
                    dest_idx = dest_point - 1
                    dest_count = board[dest_idx]
                    is_allowed_move = (dest_count * p_sign >= 0) or \
                                      (abs(dest_count) == 1)
                    if is_allowed_move:
                        moves.append((pos, dest_point))
                elif all_checkers_home:
                    if self._can_bear_off(player, pos, die_value, current_state):
                         moves.append((pos, 'off'))
        return moves

    def _get_strictly_playable_dice(self, current_state_obj, dice_list, player):
        """Determines which dice *must* or *can* be played."""
        if not dice_list: return []

        possible_moves_by_die = {}
        individually_playable_dice = []
        unique_dice = sorted(list(set(dice_list)), reverse=True)

        for die in unique_dice:
            moves = self._get_single_moves_for_die(player, die, current_state_obj)
            if moves:
                possible_moves_by_die[die] = moves
                individually_playable_dice.append(die)

        if not individually_playable_dice: return []

        is_non_double = len(dice_list) == 2 and dice_list[0] != dice_list[1]
        if is_non_double:
            smaller_die, larger_die = min(dice_list), max(dice_list)
            can_play_larger = larger_die in individually_playable_dice
            can_play_smaller = smaller_die in individually_playable_dice

            if can_play_larger and not can_play_smaller: return [larger_die]
            if not can_play_larger and can_play_smaller: return [smaller_die]

            if can_play_larger and can_play_smaller:
                can_play_both_sequence = False
                # Try large then small
                if larger_die in possible_moves_by_die:
                    for move_lg in possible_moves_by_die[larger_die]:
                        temp_game = current_state_obj.copy()
                        if temp_game.make_move_base_logic(
                            player, move_lg[0], move_lg[1]
                        ):
                            if temp_game._get_single_moves_for_die(
                                player, smaller_die, temp_game
                            ):
                                can_play_both_sequence = True; break
                # Try small then large if first failed
                if not can_play_both_sequence and \
                   smaller_die in possible_moves_by_die:
                    for move_sm in possible_moves_by_die[smaller_die]:
                        temp_game = current_state_obj.copy()
                        if temp_game.make_move_base_logic(
                            player, move_sm[0], move_sm[1]
                        ):
                            if temp_game._get_single_moves_for_die(
                                player, larger_die, temp_game
                            ):
                                can_play_both_sequence = True; break

                # If both cannot be played, but larger can, must play larger
                if not can_play_both_sequence and can_play_larger:
                    return [larger_die]
                else:
                    return individually_playable_dice
            else: return [] # Neither playable

        return individually_playable_dice

    def get_legal_actions(self):
        """Calculates all legal single moves for the current dice state."""
        player = self.current_player
        if not self.dice: return []

        playable_dice_values = self._get_strictly_playable_dice(
            self, list(self.dice), player
        )

        final_moves = set()
        for die_val in playable_dice_values:
             moves_for_this_die = self._get_single_moves_for_die(
                 player, die_val, self
             )
             final_moves.update(moves_for_this_die)

        return list(final_moves)

    def make_move(self, src, dst):
        """Applies a single validated move and updates game state."""
        player = self.current_player
        p_sign = 1 if player == 'w' else -1
        opp_bar = 'black_bar' if player == 'w' else 'white_bar'
        p_bar = 'white_bar' if player == 'w' else 'black_bar'
        p_off = 'white_off' if player == 'w' else 'black_off'

        die_to_remove = None
        nominal_die = self._get_die_for_move(src, dst, player)

        if nominal_die is not None and nominal_die in self.dice:
            die_to_remove = nominal_die
        elif dst == 'off' and isinstance(src, int):
             needed_dist = (25 - src) if player == 'w' else src
             possible_overshoot_dice = sorted(
                 [d for d in self.dice if d >= needed_dist]
             )
             for potential_die in possible_overshoot_dice:
                 if self._check_all_pieces_home(player, self) and \
                    self._can_bear_off(player, src, potential_die, self):
                      die_to_remove = potential_die
                      break

        if die_to_remove is None:
            print(f"ERROR: Cannot find die in {self.dice} for move {src}/{dst}")
            return False

        board_before = list(self.board)
        bars_before = (self.white_bar, self.black_bar)
        off_before = (self.white_off, self.black_off)
        dice_before = list(self.dice)

        try:
            if src == 'bar':
                if player == 'w': self.white_bar -= 1
                else: self.black_bar -= 1
            else:
                source_index = src - 1
                self.board[source_index] -= p_sign

            if dst == 'off':
                if player == 'w': self.white_off += 1
                else: self.black_off += 1
            else:
                dest_index = dst - 1
                dest_count = self.board[dest_index]
                if dest_count * p_sign < 0:
                    if abs(dest_count) == 1:
                        self.board[dest_index] = 0
                        if player == 'w': self.black_bar += 1
                        else: self.white_bar += 1
                        self.board[dest_index] = p_sign
                    else:
                         print(f"CRITICAL: Tried move to blocked {dst}!")
                         raise ValueError("Illegal move target")
                else:
                     self.board[dest_index] += p_sign

            self.dice.remove(die_to_remove)
            self.available_moves = self.get_legal_actions()
            game_over = self.is_game_over()

            w_final, b_final = self.get_total_checker_count()
            if w_final != 15 or b_final != 15:
                 print(f"!! CRITICAL: Count invalid after {src}/{dst} -> W:{w_final}, B:{b_final}")
                 self.winner = "ERROR"
                 return False
            return True

        except Exception as e_unexpected:
            self.board = board_before
            self.white_bar, self.black_bar = bars_before
            self.white_off, self.black_off = off_before
            self.dice = dice_before
            self.available_moves = self.get_legal_actions()
            print(f"!! UNEXPECTED Error applying make_move {player} {src}/{dst}: {e_unexpected}")
            traceback.print_exc()
            self.winner = "ERROR"
            return False

    def make_move_base_logic(self, player_base, src_base, dst_base):
        """Simplified move logic for internal AI sim (NO dice check/update)."""
        p_sign = 1 if player_base == 'w' else -1
        original_board_val_src = None
        original_board_val_dst = None
        original_bar_w, original_bar_b = self.white_bar, self.black_bar
        original_off_w, original_off_b = self.white_off, self.black_off
        src_idx, dst_idx = -1, -1

        try:
            if src_base == 'bar':
                if player_base == 'w':
                    if self.white_bar > 0: self.white_bar -= 1
                    else: return False
                else:
                    if self.black_bar > 0: self.black_bar -= 1
                    else: return False
            elif isinstance(src_base, int) and 1 <= src_base <= 24:
                src_idx = src_base - 1
                original_board_val_src = self.board[src_idx]
                if original_board_val_src * p_sign <= 0: return False
                self.board[src_idx] -= p_sign
            else: return False

            if dst_base == 'off':
                if player_base == 'w': self.white_off += 1
                else: self.black_off += 1
            elif isinstance(dst_base, int) and 1 <= dst_base <= 24:
                dst_idx = dst_base - 1
                original_board_val_dst = self.board[dst_idx]
                dest_count = original_board_val_dst

                if dest_count * p_sign < 0 and abs(dest_count) >= 2:
                    if src_idx != -1: self.board[src_idx] += p_sign # Restore source
                    return False

                if dest_count * p_sign < 0 and abs(dest_count) == 1:
                     if player_base == 'w': self.black_bar += 1
                     else: self.white_bar += 1
                     self.board[dst_idx] = 0
                     self.board[dst_idx] = p_sign
                else:
                     self.board[dst_idx] += p_sign
            else:
                if src_idx != -1: self.board[src_idx] += p_sign # Restore source
                if src_base == 'bar':
                    if player_base == 'w': self.white_bar += 1
                    else: self.black_bar += 1
                return False
            return True

        except Exception:
             if src_idx != -1 and original_board_val_src is not None:
                 self.board[src_idx] = original_board_val_src
             if dst_idx != -1 and original_board_val_dst is not None:
                 self.board[dst_idx] = original_board_val_dst
             self.white_bar, self.black_bar = original_bar_w, original_bar_b
             self.white_off, self.black_off = original_off_w, original_off_b
             return False

    def determine_game_phase(self):
        """Determines game phase based on pieces off or pip count."""
        w_home = self._check_all_pieces_home('w', self)
        b_home = self._check_all_pieces_home('b', self)

        if self.white_off > 0 or self.black_off > 0 or (w_home and b_home):
            phase = 'ENDGAME'
        else:
            w_pip = self.calculate_pip('w'); b_pip = self.calculate_pip('b');
            total_pip = w_pip + b_pip
            if total_pip > self.OPENING_EXIT_TOTAL_PIP_THRESHOLD:
                phase = 'OPENING'
            else:
                phase = 'MIDGAME'
        self.current_phase = phase
        return phase

    def roll_dice(self):
        """Rolls dice and updates internal state."""
        d1 = random.randint(1, 6)
        d2 = random.randint(1, 6)
        if d1 == d2: self.dice = [d1] * 4
        else: self.dice = [d1, d2]
        self.available_moves = self.get_legal_actions()
        return self.dice

    def switch_player(self):
        """Switches the current player and resets turn state."""
        self.current_player = 'b' if self.current_player == 'w' else 'w'
        self.dice = []
        self.available_moves = []

    def draw_board(self):
        """ Creates a text representation of the board (Your version). """
        board_template = [
            list("   13 14 15 16 17 18 |BAR| 19 20 21 22 23 24    "),
            list("   +-----------------+---+-------------------+  "),
            list("   |                 |   |                   |  "), # Row 2
            list("   |                 |   |                   |  "),
            list("   |                 |   |                   |  "),
            list("   |                 |   |                   |  "),
            list("   |                 |   |                   |  "), # Row 6
            list("   |                 |   |                   |  "),
            list("   |                 |   |                   |  "),
            list("   |                 |   |                   |  "), # Row 9
            list("   +-----------------+BAR+-------------------+  "), # Row 10
            list("   |                 |   |                   |  "), # Row 11
            list("   |                 |   |                   |  "),
            list("   |                 |   |                   |  "),
            list("   |                 |   |                   |  "),
            list("   |                 |   |                   |  "),
            list("   |                 |   |                   |  "),
            list("   |                 |   |                   |  "),
            list("   |                 |   |                   |  "), # Row 18 (Added Row)
            list("   +-----------------+---+-------------------+  "), # Row 19 (was 18)
            list("   12 11 10 09 08 07 |BAR| 06 05 04 03 02 01    ")  # Row 20 (was 19)
        ]
        num_board_rows = len(board_template)
        base_width = max(len(line) for line in board_template)
        col_map = {
             1: 43,  2: 40,  3: 37,  4: 34,  5: 31,  6: 28,
            19: 28, 20: 31, 21: 34, 22: 37, 23: 40, 24: 43,
             7: 19,  8: 16,  9: 13, 10: 10, 11:  7, 12:  4,
            13:  4, 14:  7, 15: 10, 16: 13, 17: 16, 18: 19
        }
        bar_col = 23
        off_marker_col = 1
        max_checkers_display = 5
        board_chars = [list(line) for line in board_template]

        # --- Pions sur le plateau ---
        for pos in range(1, 25):
            count = self.board[pos - 1]
            if count == 0: continue
            symbol = 'O' if count > 0 else 'X'
            abs_count = abs(count)
            col = col_map.get(pos)
            if col is None: continue

            # Top half (13-24) starts row 2, stacks down (+1)
            # Bottom half (1-12) starts row 18 (new), stacks up (-1)
            start_row = 2 if pos >= 13 else 18
            direction = 1 if pos >= 13 else -1

            for i in range(min(abs_count, max_checkers_display)):
                row_idx = start_row + (i * direction)
                if 0 <= row_idx < num_board_rows and \
                   0 <= col < len(board_chars[row_idx]) and \
                   board_template[row_idx][col] == ' ':
                    board_chars[row_idx][col] = symbol

            if abs_count > max_checkers_display:
                row_idx = start_row + ((max_checkers_display - 1) * direction)
                if 0 <= row_idx < num_board_rows and \
                   0 <= col < len(board_chars[row_idx]):
                    count_char = str(abs_count % 10)
                    board_chars[row_idx][col] = count_char

        # --- Pions sur la Barre ---
        for i in range(min(self.white_bar, max_checkers_display)):
            row_idx = 2 + i
            if 0 <= row_idx < num_board_rows and \
               0 <= bar_col < len(board_chars[row_idx]) and \
               board_template[row_idx][bar_col] == ' ':
                 board_chars[row_idx][bar_col] = 'O'
        if self.white_bar > max_checkers_display:
             row_idx = 2 + max_checkers_display - 1
             if 0 <= row_idx < num_board_rows and \
                0 <= bar_col < len(board_chars[row_idx]):
                count_char = str(self.white_bar % 10)
                board_chars[row_idx][bar_col] = count_char

        for i in range(min(self.black_bar, max_checkers_display)):
            # Start from the new bottom row (18) and go up
            row_idx = 18 - i
            if 0 <= row_idx < num_board_rows and \
               0 <= bar_col < len(board_chars[row_idx]) and \
               board_template[row_idx][bar_col] == ' ':
                 board_chars[row_idx][bar_col] = 'X'
        if self.black_bar > max_checkers_display:
             row_idx = 18 - (max_checkers_display - 1)
             if 0 <= row_idx < num_board_rows and \
                0 <= bar_col < len(board_chars[row_idx]):
                count_char = str(self.black_bar % 10)
                board_chars[row_idx][bar_col] = count_char

        # --- Pions Sortis (Off) ---
        for r_clear in range(2, 19): # Adjust range for new row
            if 0 <= r_clear < num_board_rows and \
               0 <= off_marker_col < len(board_chars[r_clear]):
                 if board_template[r_clear][off_marker_col] == ' ':
                     board_chars[r_clear][off_marker_col] = ' '

        for i in range(min(self.white_off, max_checkers_display)):
            row = 2 + i
            if 0 <= row < num_board_rows and \
               0 <= off_marker_col < len(board_chars[row]):
                 board_chars[row][off_marker_col] = 'O'
        if self.white_off > max_checkers_display:
            row = 2 + max_checkers_display - 1
            target_col = off_marker_col
            disp_val = str(self.white_off % 10)
            if 0 <= row < num_board_rows and \
               0 <= target_col < len(board_chars[row]):
                 board_chars[row][target_col] = disp_val

        for i in range(min(self.black_off, max_checkers_display)):
            row = 18 - i # Adjust start row
            if 0 <= row < num_board_rows and \
               0 <= off_marker_col < len(board_chars[row]):
                 board_chars[row][off_marker_col] = 'X'
        if self.black_off > max_checkers_display:
            row = 18 - (max_checkers_display - 1) # Adjust start row
            target_col = off_marker_col
            disp_val = str(self.black_off % 10)
            if 0 <= row < num_board_rows and \
               0 <= target_col < len(board_chars[row]):
                 board_chars[row][target_col] = disp_val

        # --- Infos sur le côté droit ---
        dice_faces_large = {
            1: ["+-------+", "|       |", "|   o   |", "|       |", "+-------+"],
            2: ["+-------+", "| o     |", "|       |", "|     o |", "+-------+"],
            3: ["+-------+", "| o     |", "|   o   |", "|     o |", "+-------+"],
            4: ["+-------+", "| o   o |", "|       |", "| o   o |", "+-------+"],
            5: ["+-------+", "| o   o |", "|   o   |", "| o   o |", "+-------+"],
            6: ["+-------+", "| o o o |", "|       |", "| o o o |", "+-------+"]
        }
        dice_height = 5; dice_width = 9; side_info_col = base_width + 1

        def write_text(text, row_idx, col, max_width=None):
             if not (0 <= row_idx < num_board_rows): return
             if max_width: text = text[:max_width]
             needed_len = col + len(text); current_len = len(board_chars[row_idx])
             if current_len < needed_len:
                 board_chars[row_idx].extend([' '] * (needed_len - current_len))
             for i, char in enumerate(text):
                 target_col = col + i
                 if 0 <= target_col < len(board_chars[row_idx]):
                     board_chars[row_idx][target_col] = char

        opponent = 'b' if self.current_player == 'w' else 'w'
        last_sequence = self.black_last_turn_sequence if opponent == 'b' \
            else self.white_last_turn_sequence
        opp_last_move_str = f"{opponent.upper()} Last:"
        if last_sequence:
            formatted_seq_parts = [f"{m[0]}/{m[1]}" for m in last_sequence]
            formatted_seq = ", ".join(formatted_seq_parts)
            opp_last_move_str += f" {formatted_seq}"
        else: opp_last_move_str += " -"
        write_text(opp_last_move_str, 2, side_info_col, max_width=30)

        player_symbol = 'O' if self.current_player == 'w' else 'X'
        phase_str = (
            f"Turn: {self.current_player.upper()}[{player_symbol}] "
            f"Ph:{self.current_phase}"
        )
        write_text(phase_str, 3, side_info_col, max_width=30)

        write_text(f"W Off [O]: {self.white_off: >2}", 4, side_info_col, max_width=15)
        # Adjust row for Black Off count due to added row in template
        write_text(f"B Off [X]: {self.black_off: >2}", 17, side_info_col, max_width=15) # Was 16

        die1_base_row, die2_base_row = 5, 11
        dice_to_draw = list(self.dice)
        dice_drawn_count = 0
        dice_values_to_draw = [None, None]
        if dice_to_draw:
            dice_values_to_draw[0] = dice_to_draw.pop(0); dice_drawn_count += 1
        if dice_to_draw:
            dice_values_to_draw[1] = dice_to_draw.pop(0); dice_drawn_count += 1

        for idx, base_row in enumerate([die1_base_row, die2_base_row]):
            die_val = dice_values_to_draw[idx]
            if die_val is not None and die_val in dice_faces_large:
                face = dice_faces_large[die_val]
                for i in range(dice_height):
                    write_text(face[i], base_row + i, side_info_col, max_width=dice_width)
            else:
                for i in range(dice_height):
                    write_text(" " * dice_width, base_row + i, side_info_col, max_width=dice_width)

        # Adjust row for remaining dice display
        more_dice_row = 18 # Was 17
        write_text(" " * 20, more_dice_row, side_info_col, max_width=20)
        if len(self.dice) > dice_drawn_count:
            remaining_dice_str = ', '.join(map(str, self.dice[dice_drawn_count:]))
            extra_dice_str = f"({remaining_dice_str} left)"
            write_text(extra_dice_str, more_dice_row, side_info_col, max_width=20)

        board_lines = [''.join(row_chars).rstrip() for row_chars in board_chars]
        return '\n'.join(board_lines)

    # ============================================================
    # ================= END OF draw_board ========================
    # ============================================================


    def evaluate_position_heuristic(self, game_state, player_to_evaluate, weights: HeuristicWeights):
        opp='b' if player_to_evaluate=='w' else 'w';p_sign=1 if player_to_evaluate=='w' else -1;o_sign=-p_sign;board=game_state.board
        p_pip=game_state.calculate_pip(player_to_evaluate);o_pip=game_state.calculate_pip(opp);pip_score=(o_pip-p_pip)*weights.PIP_SCORE_FACTOR
        p_off=game_state.white_off if player_to_evaluate=='w' else game_state.black_off;o_off=game_state.black_off if player_to_evaluate=='w' else game_state.white_off;off_score=(p_off-o_off)*weights.OFF_SCORE_FACTOR
        p_bar=game_state.white_bar if player_to_evaluate=='w' else game_state.black_bar;o_bar=game_state.black_bar if player_to_evaluate=='w' else game_state.white_bar;bar_penalty=p_bar*weights.BAR_PENALTY;hit_bonus=o_bar*weights.HIT_BONUS
        point_bonus_total=0.0;home_point_bonus_total=0.0;inner_home_bonus_total=0.0;anchor_bonus_total=0.0;blot_penalty_total=0.0;trapped_checker_bonus_total=0.0;made_points_mask=[0]*24;player_blot_positions=[]
        for i in range(24):
            pos=i+1;count=board[i];player_checker_count=count*p_sign
            if player_checker_count>=2:
                made_points_mask[i]=1;point_bonus_total+=weights.POINT_BONUS
                is_home=(player_to_evaluate=='w' and 19<=pos<=24) or (player_to_evaluate=='b' and 1<=pos<=6)
                if is_home:home_point_bonus_total+=weights.HOME_BOARD_POINT_BONUS
                is_inner_home=(player_to_evaluate=='w' and 22<=pos<=24) or (player_to_evaluate=='b' and 1<=pos<=3)
                if is_inner_home:inner_home_bonus_total+=weights.INNER_HOME_POINT_BONUS
                is_anchor=(player_to_evaluate=='w' and 1<=pos<=6) or (player_to_evaluate=='b' and 19<=pos<=24)
                if is_anchor:anchor_bonus_total+=weights.ANCHOR_BONUS
            elif player_checker_count==1:player_blot_positions.append(i)
        if player_blot_positions:
            opponent_checker_indices=set()
            for idx,count in enumerate(board):
                if count*o_sign>0: opponent_checker_indices.add(idx)
            for blot_idx in player_blot_positions:
                direct_shots=0
                for shot_dist in range(1,7):
                    shooter_idx=blot_idx-shot_dist*o_sign
                    if 0<=shooter_idx<24 and shooter_idx in opponent_checker_indices:
                        direct_shots+=abs(board[shooter_idx])
                if o_bar>0:
                    entry_die_needed=(blot_idx+1) if player_to_evaluate=='b' else (24-blot_idx)
                    if 1<=entry_die_needed<=6:
                        opp_entry_point_idx=(entry_die_needed-1) if player_to_evaluate=='b' else (24-entry_die_needed)
                        if board[opp_entry_point_idx]*p_sign<2: direct_shots+=o_bar
                penalty_for_this_blot=direct_shots*weights.DIRECT_SHOT_PENALTY_FACTOR;blot_penalty_total+=penalty_for_this_blot
            if o_bar>0: blot_penalty_total*=weights.BLOT_PENALTY_REDUCTION_IF_OPP_ON_BAR
        prime_bonus_total=0.0;max_prime_len=0;current_prime_len=0;prime_segments=[]
        for i in range(24):
            if made_points_mask[i]==1: current_prime_len+=1
            else:
                if current_prime_len>=4:
                    prime_end_idx=i-1;prime_start_idx=prime_end_idx-current_prime_len+1
                    prime_segments.append({'start':prime_start_idx,'end':prime_end_idx,'len':current_prime_len})
                    prime_bonus_total+=(current_prime_len-3)*weights.PRIME_BASE_BONUS
                max_prime_len=max(max_prime_len,current_prime_len);current_prime_len=0
        if current_prime_len>=4:
            prime_end_idx=23;prime_start_idx=prime_end_idx-current_prime_len+1
            prime_segments.append({'start':prime_start_idx,'end':prime_end_idx,'len':current_prime_len})
            prime_bonus_total+=(current_prime_len-3)*weights.PRIME_BASE_BONUS
        max_prime_len=max(max_prime_len,current_prime_len)
        if weights.TRAPPED_CHECKER_BONUS!=0:
            for prime in prime_segments:
                if prime['len']>=5:
                    trapped_count=0
                    trap_zone_indices=range(prime['start']) if player_to_evaluate=='w' else range(prime['end']+1,24)
                    for trap_idx in trap_zone_indices:
                        if board[trap_idx]*o_sign>0: trapped_count+=abs(board[trap_idx])
                    trapped_checker_bonus_total+=trapped_count*weights.TRAPPED_CHECKER_BONUS
        midgame_prison_bonus=0.0;home_points_made_count=0
        home_range=range(18,24) if player_to_evaluate=='w' else range(6)
        for i in home_range:
            if made_points_mask[i]==1: home_points_made_count+=1
        if hasattr(weights,'MIDGAME_HOME_PRISON_BONUS') and weights.MIDGAME_HOME_PRISON_BONUS!=0 and home_points_made_count>=3 and o_bar>0:
             midgame_prison_bonus=weights.MIDGAME_HOME_PRISON_BONUS*o_bar
        back_checker_penalty=0.0;is_far_behind=p_pip>0 and o_pip>0 and p_pip>=1.5*o_pip
        if hasattr(weights,'FAR_BEHIND_BACK_CHECKER_PENALTY_FACTOR') and weights.FAR_BEHIND_BACK_CHECKER_PENALTY_FACTOR!=0 and is_far_behind:
            back_checker_pip_sum=0
            back_zone=range(1,7) if player_to_evaluate=='w' else range(19,25)
            for pos in back_zone:
                point_index=pos-1; count=game_state.board[point_index]
                if count*p_sign>0:
                    distance=(25-pos) if player_to_evaluate=='w' else pos
                    back_checker_pip_sum+=distance*abs(count)
            back_checker_penalty=back_checker_pip_sum*weights.FAR_BEHIND_BACK_CHECKER_PENALTY_FACTOR*-1.0
        total_score=(pip_score+off_score+bar_penalty+hit_bonus+point_bonus_total+home_point_bonus_total+inner_home_bonus_total+anchor_bonus_total+prime_bonus_total+blot_penalty_total+midgame_prison_bonus+trapped_checker_bonus_total+back_checker_penalty)
        return total_score

# --- AI Functions ---
def _get_random_dice_sample():
    """Generates NUM_DICE_SAMPLES random dice pairs."""
    samples = []
    for _ in range(NUM_DICE_SAMPLES):
        d1 = random.randint(1, 6)
        d2 = random.randint(1, 6)
        samples.append((d1, d2))
    return samples

def generate_possible_next_states_with_sequences(
        current_game_state: BackgammonGame,
        dice_tuple: tuple,
        player: str
    ):
    """
    Generates pairs of (final_state, sequence) for all unique reachable states
    after playing the full dice roll, enforcing max dice played rule.
    """
    possible_final_outcomes = {} # Stores { board_tuple : (state, sequence) }
    memo = {} # Memoization for recursive calls (state_tuple, dice_rem_tuple)

    # --- Inner recursive function ---
    def find_sequences_recursive(
            state_now: BackgammonGame,
            dice_rem_tuple: tuple,
            sequence_so_far: list
        ):
        """Explores move sequences recursively."""
        state_key_memo = (state_now.board_tuple(), dice_rem_tuple)
        if state_key_memo in memo:
            return # Already explored this exact situation
        memo[state_key_memo] = True

        # Key for storing results: just the board state
        current_board_key = state_now.board_tuple()

        # --- Base case: No dice left ---
        if not dice_rem_tuple:
            # Store if this final state hasn't been reached before,
            # or if this path is shorter (shouldn't happen with correct logic but safe).
            # We store a copy of the state and the sequence that led to it.
            if current_board_key not in possible_final_outcomes or len(sequence_so_far) < len(possible_final_outcomes[current_board_key][1]):
                 possible_final_outcomes[current_board_key] = (state_now.copy(), list(sequence_so_far))
            return

        # --- Recursive Step ---
        # Determine which dice can actually be played from this state_now
        # Must pass state_now explicitly as we are inside recursion
        allowed_next_dice = state_now._get_strictly_playable_dice(state_now, list(dice_rem_tuple), player)

        # If no dice can be played, this sequence ends here. Store the current state.
        if not allowed_next_dice:
            if current_board_key not in possible_final_outcomes or len(sequence_so_far) < len(possible_final_outcomes[current_board_key][1]):
                 possible_final_outcomes[current_board_key] = (state_now.copy(), list(sequence_so_far))
            return

        # Explore moves for each playable die value
        found_at_least_one_move_this_level = False
        for die_val in allowed_next_dice:
            # Find all single moves possible with this die from state_now
            possible_single_moves = state_now._get_single_moves_for_die(player, die_val, state_now)

            if possible_single_moves:
                found_at_least_one_move_this_level = True
                for move in possible_single_moves:
                    src, dst = move
                    # --- Figure out which die was *actually* consumed ---
                    consumed_die = None
                    temp_dice_list = list(dice_rem_tuple) # Available dice for this path
                    nominal_die = state_now._get_die_for_move(src, dst, player)

                    # Case 1: Nominal die matches current die value and is available
                    if nominal_die is not None and nominal_die == die_val and die_val in temp_dice_list:
                         consumed_die = die_val
                    # Case 2: Bear-off overshoot check
                    elif dst == 'off' and isinstance(src, int):
                        needed_dist = (25 - src) if player == 'w' else src
                        # Check if the current die_val *could* perform the bear off
                        if die_val >= needed_dist and state_now._can_bear_off(player, src, die_val, state_now) and die_val in temp_dice_list:
                            # Find the *smallest available* die that also works
                            valid_overshoot_dice = sorted([
                                d for d in temp_dice_list if d >= needed_dist and
                                state_now._can_bear_off(player, src, d, state_now)
                            ])
                            if valid_overshoot_dice:
                                smallest_valid = valid_overshoot_dice[0]
                                # Consume the die only if the current die_val *is* that smallest valid die
                                if die_val == smallest_valid:
                                     consumed_die = die_val

                    # If no die could be validly consumed for this move/die_val combo, skip
                    if consumed_die is None:
                        continue

                    # --- Simulate the move ---
                    next_state = state_now.copy() # Create a copy for the next step
                    # Use base logic (no dice update) for simulation
                    move_ok = next_state.make_move_base_logic(player, src, dst)

                    if move_ok:
                        # Prepare remaining dice for the recursive call
                        next_dice_list = list(dice_rem_tuple)
                        try:
                            next_dice_list.remove(consumed_die) # Consume the die
                        except ValueError:
                             continue
                        next_dice_rem_tuple = tuple(sorted(next_dice_list))

                        # Add move to current path and recurse
                        sequence_so_far.append(move)
                        find_sequences_recursive(next_state, next_dice_rem_tuple, sequence_so_far)
                        sequence_so_far.pop() # Backtrack: remove move after exploring subtree

        # If no single move was possible *for any* of the allowed dice
        if not found_at_least_one_move_this_level:
             if current_board_key not in possible_final_outcomes or len(sequence_so_far) < len(possible_final_outcomes[current_board_key][1]):
                 possible_final_outcomes[current_board_key] = (state_now.copy(), list(sequence_so_far))
    # --- End of Inner Recursive Function ---

    # --- Main part of generate_possible_next_states_with_sequences ---
    if not dice_tuple: return [(current_game_state.copy(), [])]

    if len(dice_tuple) == 4: initial_dice_perms = [tuple(dice_tuple)]
    elif len(dice_tuple) == 2:
         d1, d2 = dice_tuple[0], dice_tuple[1]
         initial_dice_perms = [(d1, d2), (d2, d1)] if d1 != d2 else [(d1, d1)] # Handle case like (3,3) input
    else: initial_dice_perms = [dice_tuple]

    for initial_dice in initial_dice_perms:
        search_start_state = current_game_state.copy()
        search_start_state.dice = list(initial_dice)
        find_sequences_recursive(search_start_state, tuple(sorted(initial_dice)), [])

    if not possible_final_outcomes: return [(current_game_state.copy(), [])]

    # Filter for max dice played rule
    max_len = 0
    if possible_final_outcomes:
         try: max_len = max(len(seq) for _, seq in possible_final_outcomes.values())
         except ValueError: max_len = 0

    final_results = []
    if max_len == 0 and len(possible_final_outcomes) >= 1:
         first_outcome = list(possible_final_outcomes.values())[0]
         final_results = [first_outcome]
    else:
         for outcome in possible_final_outcomes.values():
             state_obj, seq_list = outcome
             if len(seq_list) == max_len: final_results.append(outcome)

    if not final_results and possible_final_outcomes:
        final_results = list(possible_final_outcomes.values())

    return final_results # List of (final_state_object, sequence_list)


def get_minimax_score_sampled(
        game_state: BackgammonGame, current_turn_player: str, depth: int,
        maximizing_player: str, alpha: float, beta: float):
    """Minimax function with alpha-beta pruning and dice sampling."""
    if game_state.is_game_over():
        if game_state.winner == maximizing_player: return float('inf')
        elif game_state.winner is not None: return float('-inf')
        else: return 0
    if depth == 0:
        phase = game_state.determine_game_phase()
        weights = PHASE_WEIGHTS.get(phase, MIDGAME_WEIGHTS)
        score = game_state.evaluate_position_heuristic(game_state, maximizing_player, weights)
        return score

    is_maximizing_node = (current_turn_player == maximizing_player)
    opponent_player = 'b' if current_turn_player == 'w' else 'w'
    sampled_dice_rolls = _get_random_dice_sample()
    accumulated_score = 0.0
    num_samples_processed = 0

    if is_maximizing_node:
        expected_value = 0.0
        for d1, d2 in sampled_dice_rolls:
            dice_for_turn = (d1, d2) * 2 if d1 == d2 else (d1, d2)
            possible_outcomes = generate_possible_next_states_with_sequences(
                game_state, dice_for_turn, current_turn_player)
            best_eval_for_this_roll = float('-inf')
            no_move_outcome = (not possible_outcomes or (len(possible_outcomes) == 1 and not possible_outcomes[0][1]))
            if no_move_outcome:
                 best_eval_for_this_roll = get_minimax_score_sampled(
                     game_state, opponent_player, depth - 1, maximizing_player, alpha, beta)
            else:
                for next_state, _ in possible_outcomes:
                    evaluation = get_minimax_score_sampled(
                        next_state, opponent_player, depth - 1, maximizing_player, alpha, beta)
                    best_eval_for_this_roll = max(best_eval_for_this_roll, evaluation)
                    alpha = max(alpha, best_eval_for_this_roll)
                    if beta <= alpha: break
            accumulated_score += best_eval_for_this_roll; num_samples_processed += 1
        avg_score = accumulated_score / num_samples_processed if num_samples_processed > 0 else 0.0
        return avg_score
    else: # Minimizing node
        expected_value = 0.0
        for d1, d2 in sampled_dice_rolls:
            dice_for_turn = (d1, d2) * 2 if d1 == d2 else (d1, d2)
            possible_outcomes = generate_possible_next_states_with_sequences(
                game_state, dice_for_turn, current_turn_player)
            worst_eval_for_this_roll = float('inf')
            no_move_outcome = (not possible_outcomes or (len(possible_outcomes) == 1 and not possible_outcomes[0][1]))
            if no_move_outcome:
                 worst_eval_for_this_roll = get_minimax_score_sampled(
                     game_state, opponent_player, depth - 1, maximizing_player, alpha, beta)
            else:
                for next_state, _ in possible_outcomes:
                    evaluation = get_minimax_score_sampled(
                        next_state, opponent_player, depth - 1, maximizing_player, alpha, beta)
                    worst_eval_for_this_roll = min(worst_eval_for_this_roll, evaluation)
                    beta = min(beta, worst_eval_for_this_roll)
                    if beta <= alpha: break
            accumulated_score += worst_eval_for_this_roll; num_samples_processed += 1
        avg_score = accumulated_score / num_samples_processed if num_samples_processed > 0 else 0.0
        return avg_score


def select_ai_move_minimax(
        current_game_state: BackgammonGame, dice_tuple: tuple, ai_player: str):
    """Selects best move sequence and resulting state for AI using Minimax."""
    possible_outcomes = generate_possible_next_states_with_sequences(
        current_game_state, dice_tuple, ai_player)

    if not possible_outcomes or (len(possible_outcomes) == 1 and not possible_outcomes[0][1]):
        return [], current_game_state # No move possible

    best_score = float('-inf')
    optimal_resulting_state = None
    optimal_sequence = []
    opponent_player = 'b' if ai_player == 'w' else 'w'

    for next_state, sequence in possible_outcomes:
        score_for_state = get_minimax_score_sampled(
            next_state, opponent_player, MAX_DEPTH - 1, ai_player,
            float('-inf'), float('inf'))

        if score_for_state > best_score:
            best_score = score_for_state
            optimal_resulting_state = next_state
            optimal_sequence = sequence

    if optimal_resulting_state is None:
        if possible_outcomes:
             optimal_resulting_state = possible_outcomes[0][0]
             optimal_sequence = possible_outcomes[0][1]
        else:
             optimal_resulting_state = current_game_state
             optimal_sequence = []

    # Ensure dice/moves are cleared in the returned state as AI turn is done
    if optimal_resulting_state:
        optimal_resulting_state.dice = []
        optimal_resulting_state.available_moves = []

    return optimal_sequence, optimal_resulting_state


# --- Main Game Loop ---
def main_play_vs_ai():
    """Plays Human vs Hybrid AI."""
    print("\n" + "=" * 70)
    print(" Play Backgammon vs Hybrid AI (Adaptive Heuristic + 3-Ply Minimax)")
    print("=" * 70)

    human_player = None
    while human_player not in ['w', 'b']:
        choice = input("Play as White ('w'=O, starts) or Black ('b'=X)? ").lower().strip()
        human_player = choice if choice in ['w', 'b'] else None

    ai_player = 'b' if human_player == 'w' else 'w'
    print(f"\nOkay, you are Player {human_player.upper()} "
          f"({'O' if human_player == 'w' else 'X'}). AI is {ai_player.upper()}.")
    print(f"AI Search Depth: {MAX_DEPTH}, Dice Samples: {NUM_DICE_SAMPLES}")
    print("Enter moves as 'src/dst' (e.g., 13/7, bar/5, 24/off).")
    input("Press Enter to start...")

    game = BackgammonGame(human_player=human_player)
    game.current_player = 'w'
    turn_count = 0

    while game.winner is None:
        turn_count += 1
        is_human_turn = (game.current_player == human_player)
        player_name = "Your Turn" if is_human_turn else "AI's Turn"
        player_symbol = 'O' if game.current_player == 'w' else 'X'

        current_phase = game.determine_game_phase()
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n" + "=" * 25 +
              f" Turn {turn_count}: {player_name} ({player_symbol}) " +
              "=" * 25)

        # Roll dice & update available moves for the current player
        current_dice_roll = game.roll_dice()

        # Display board and game info
        print(game.draw_board())
        pip_h = game.calculate_pip(human_player)
        pip_ai = game.calculate_pip(ai_player)
        pip_diff = pip_h - pip_ai
        print(f"\n   Phase: {current_phase}")
        print(f"   Your Pips: {pip_h} | AI Pips: {pip_ai} | "
              f"Diff: {pip_diff:+} (Lower is better)")
        print(f"   Bar: W={game.white_bar} B={game.black_bar} | "
              f"Off: W={game.white_off} B={game.black_off}")

        played_sequence_this_turn = [] # Reset sequence for this turn

        # --- Check if moves are possible ---
        if not game.available_moves:
            print("\n   No legal moves! Turn skipped.")
            time.sleep(2.0)
        else:
            # --- Handle Player's Turn ---
            if is_human_turn:
                moves_made_count = 0
                original_dice_for_turn = list(game.dice)
                max_moves_possible = len(original_dice_for_turn)

                while game.dice and game.available_moves:
                    print("\n" + '-' * 40)
                    print(f"   Dice left: {game.dice}")
                    sorted_available = sorted(
                        game.available_moves,
                        key=lambda m: (str(m[0]), str(m[1]))
                    )
                    print(f"   Available moves: {sorted_available}")

                    move_prompt = (
                        f"   Enter move {moves_made_count + 1}/"
                        f"{max_moves_possible} (or 'p' to pass): "
                    )
                    move_input = input(move_prompt).lower().strip()

                    if move_input == 'p':
                        print("   Passing remaining moves...")
                        time.sleep(1)
                        break

                    _, src, dst = game.parse_move(move_input, game.current_player)
                    if src is None or dst is None:
                        print("   Invalid move format. Use 'src/dst'.")
                        continue

                    current_move_tuple = (src, dst)
                    if current_move_tuple in game.available_moves:
                        move_successful = game.make_move(src, dst)
                        if move_successful:
                            played_sequence_this_turn.append(current_move_tuple)
                            moves_made_count += 1
                            current_phase = game.determine_game_phase()
                            os.system('cls' if os.name == 'nt' else 'clear')
                            print("\n" + "=" * 25 +
                                  f" Turn {turn_count}: Your Turn ({player_symbol})"
                                  f" - Move {moves_made_count} " + "=" * 25)

                            # Update display sequence during turn for immediate feedback
                            if game.current_player == 'w':
                                game.white_last_turn_sequence = \
                                    played_sequence_this_turn
                            else:
                                game.black_last_turn_sequence = \
                                    played_sequence_this_turn

                            print(game.draw_board()) # Redraw
                            pip_h = game.calculate_pip(human_player)
                            pip_ai = game.calculate_pip(ai_player)
                            pip_diff = pip_h - pip_ai
                            print(f"\n   Phase: {current_phase}")
                            print(f"   Your Pips: {pip_h} | AI Pips: {pip_ai} | "
                                  f"Diff: {pip_diff:+}")
                            print(f"   Bar: W={game.white_bar} B={game.black_bar} | "
                                  f"Off: W={game.white_off} B={game.black_off}")

                            if game.winner: break
                        else:
                            print(f"   ERROR: Move {src}/{dst} failed?")
                            time.sleep(2)
                    else:
                        print(f"   Invalid move: '{move_input}' not available.")
                        time.sleep(1)

                if not game.winner:
                    print("\n   End of your turn.")

            # --- Handle AI's Turn ---
            else:
                print(f"\n   AI ({player_symbol}) is thinking...")
                ai_start_time = time.time()
                dice_for_ai = tuple(game.dice)

                chosen_sequence, best_resulting_state = select_ai_move_minimax(
                    game, dice_for_ai, ai_player
                )
                ai_end_time = time.time()

                # Update the main game state object
                game = best_resulting_state
                played_sequence_this_turn = chosen_sequence

                # Display results
                current_phase = game.determine_game_phase()
                os.system('cls' if os.name == 'nt' else 'clear')
                print("\n" + "=" * 25 +
                      f" Turn {turn_count}: AI's Turn ({player_symbol}) - Completed " +
                      "=" * 25)
                print(f"\n   Original Roll: {dice_for_ai}")
                print(f"   AI Calculation Time: {ai_end_time - ai_start_time:.2f}s")
                if played_sequence_this_turn:
                    formatted_seq_parts = [f"{m[0]}/{m[1]}" for m in played_sequence_this_turn]
                    formatted_seq = ", ".join(formatted_seq_parts)
                    print(f"   AI Played: {formatted_seq}")
                else:
                     print("   AI made no move or sequence not captured.")

                print(game.draw_board()) # Display board after AI move
                pip_h = game.calculate_pip(human_player)
                pip_ai = game.calculate_pip(ai_player)
                pip_diff = pip_h - pip_ai
                print(f"\n   Phase: {current_phase}")
                print(f"   Your Pips: {pip_h} | AI Pips: {pip_ai} | "
                      f"Diff: {pip_diff:+}")
                print(f"   Bar: W={game.white_bar} B={game.black_bar} | "
                      f"Off: W={game.white_off} B={game.black_off}")

                if not game.winner:
                    print("\n   End of AI's turn.")
                    time.sleep(1.0)

        # --- End of Turn Logic ---


        if game.current_player == 'w':
             game.white_last_turn_sequence = played_sequence_this_turn
             game.black_last_turn_sequence = [] # Clear opponent's history
        else:
             game.black_last_turn_sequence = played_sequence_this_turn
             game.white_last_turn_sequence = []

        if game.winner:
            break


        game.switch_player()



    # --- Game End Display ---
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\n" + "=" * 35 + " GAME OVER " + "=" * 35)
    print(f"   Game lasted {turn_count} turns.")
    print("\n--- Final Board State ---")
    print(game.draw_board()) # Show final board
    pip_h = game.calculate_pip(human_player)
    pip_ai = game.calculate_pip(ai_player)
    pip_diff = pip_h - pip_ai
    print(f"\n   Final Bar: W={game.white_bar} B={game.black_bar}")
    print(f"   Final Off: W={game.white_off} B={game.black_off}")
    print(f"   Final Pips: You={pip_h}  AI={pip_ai}  (Difference: {pip_diff:+})")
    print("\n" + "-" * 80)

    if game.winner == human_player:
        print(f"   CONGRATULATIONS! You ({game.winner.upper()}) won!")
    elif game.winner == ai_player:
        print(f"   Sorry! The AI ({game.winner.upper()}) won.")
    elif game.winner == "DRAW":
        print("   It's a DRAW!")
    elif game.winner == "ERROR":
        print("   Game ended due to an internal ERROR.")
    else:
        print(f"   Game ended with unexpected status: {game.winner}")

    print("-" * 80 + "\n")
    input("Press Enter to exit.")



if __name__ == "__main__":
    main_play_vs_ai()
