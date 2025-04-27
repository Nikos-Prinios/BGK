# BGK
A sweet ASCII backgammon game (Minimax + Adaptive Heuristic)

## Goal

This project aims to create a strong Artificial Intelligence (AI) player for the game of Backgammon, designed to be a challenging opponent for a human player via a command-line (ASCII) interface.

It represents a **hybrid approach**, combining a proven search structure with a sophisticated evaluation function derived from a previous personnal project.

## Key Features

*   **Hybrid AI:** The core of this project is the fusion of two approaches:
    *   **Minimax Search (3-Ply):** Uses the Minimax algorithm to explore possible moves up to a depth of 3 half-turns (3-ply), allowing the AI to anticipate potential responses. The fundamental logic for this search is inspired by and adapted from the [llaki/BackgammonAI](https://github.com/llaki/BackgammonAI/tree/master) project.
    *   **Advanced Adaptive Heuristic:** The position evaluation function (`evaluate_position_heuristic`) comes from a previous script (`BKG-21_04.py`). It is significantly more detailed than basic evaluations and, crucially, **adapts its weights** based on the detected game phase (Opening, Midgame, Endgame). It considers numerous strategic factors:
        *   Pip Count difference (the race)
        *   Checkers borne off (Off Score)
        *   Hit checkers and checkers on the bar (Hit Bonus / Bar Penalty)
        *   Points made (anchors, blocks)
        *   Points in the home board (strategically important)
        *   Primes (sequences of blocking points)
        *   Exposed single checkers ("blots") and their vulnerability to direct shots
        *   Opponent's checkers trapped behind a prime
        *   Situational bonuses/penalties (e.g., opponent on the bar, player significantly behind)

*   **Alpha-Beta Pruning:** The Minimax search is optimized using Alpha-Beta pruning, significantly reducing the number of nodes to explore without affecting the final result for a given depth.

*   **Dice Sampling:** To further speed up the search in "chance" nodes (opponent's dice rolls within the Minimax tree), the AI does not evaluate *all* 21 distinct dice combinations. Instead, it evaluates a random sample of `NUM_DICE_SAMPLES` (default 14) dice rolls and calculates an expected score based on this sample. This allows reaching a 3-ply depth in reasonable time.

*   **ASCII Interface:** The game is played entirely in the terminal using a clear text-based interface, displaying the board, dice, pips, and turn information. Because it feels good.

*   **Player vs AI Mode:** The script is specifically designed for a human to play against this hybrid AI.

## How It Works (AI Decision Making)

1.  **Dice Roll:** The AI receives its dice roll for the turn.
2.  **Generate Successor States:** The script generates all *legal and complete* move sequences possible for that dice roll, respecting the rules (play max dice, play higher die if blocked, etc.). It produces a list of `(final_board_state, move_sequence)` pairs.
3.  **Minimax Evaluation:** For each possible `final_board_state`:
    *   The AI initiates a Minimax search (depth `MAX_DEPTH - 1`) simulating the opponent's reply.
    *   Chance nodes (simulated opponent rolls) use dice sampling.
    *   Alpha-Beta pruning is applied.
    *   At the leaves of the search tree (depth 0 or game over), the `evaluate_position_heuristic` function (with phase-adapted weights) is called to get a score.
4.  **Move Selection:** The AI chooses the `move_sequence` that leads to the `final_board_state` which received the highest score during the Minimax evaluation.
5.  **Update:** The main game state is updated to reflect the board after executing the AI's chosen sequence.

## How to Play

1.  **Prerequisites:** Ensure you have Python 3 installed.
2.  **Run the Script:** Open a terminal or command prompt, navigate to the directory containing the script, and run:
    ```bash
    python bkg.py
    ```

3.  **Choose Color:** The script will ask if you want to play as White ('w', starts first) or Black ('b').
4.  **Play:**
    *   The board will be displayed.
    *   When it's your turn, the dice are rolled, and legal moves are listed.
    *   Enter your moves in the format `source/destination` (e.g., `13/7`, `bar/5`, `24/off`).
    *   You can type `p` to pass if you cannot or choose not to play any more dice for the turn.
    *   The AI will automatically play its turn.

## Acknowledgments and Credits

This project merges two approaches:

*   The **Minimax search structure with Alpha-Beta pruning and dice sampling** was inspired by and adapted from the java code provided by Llasserre / llaki in the [llaki/BackgammonAI](https://github.com/llaki/BackgammonAI/tree/master) project. Thanks for that clear and effective initial work.
*   The **detailed adaptive heuristic evaluation function** and the **ASCII game interface** originate from a previous personal script (`BKG-21_04.py`).
*   (+) An unvaluable help has been provided by Google AI Studio.

The goal was to leverage the predictive power of Minimax while benefiting from the nuances of an advanced, dynamic position evaluation.

## Potential Improvements

*   **Opening Book:** Integrate a database of standard strong opening moves to speed up and solidify the early game.
*   **Adaptive Search Depth:** Adjust `MAX_DEPTH` based on the game phase or position complexity (e.g., deeper in tactical middlegames, shallower in pure races).
*   **Optimization:** Profile the code to identify potential bottlenecks (e.g., state generation, evaluation).
*   **AI Sequence Display:** Implement a method to clearly deduce and display the exact sequence played by the AI within `draw_board`.
