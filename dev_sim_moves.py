#!/usr/bin/env python3
"""Cycle the robot (real or simulated) through every move in moves.py.

For watching and tuning choreography without going through the whole
question flow. Run the daemon first, then:

    .venv/bin/python dev_sim_moves.py            # all moves, 6s each
    .venv/bin/python dev_sim_moves.py spin_up    # loop one move until Ctrl-C
"""

import sys
import time

from reachy_mini import ReachyMini

from reachy_search import moves

ALL = {
    "idle": moves.idle,
    "listening": moves.listening,
    "attentive": moves.attentive,
    **moves.THINKING_MOVES,
    **moves.BENCH_MOVES,
    "speaking": moves.speaking,
    "error_shake": moves.error_shake,
    "greet": moves.greet,
}


def play(mini: ReachyMini, name: str, seconds: float | None) -> None:
    move = ALL[name]
    print(f"--- {name} ---")
    t0 = time.monotonic()
    while seconds is None or time.monotonic() - t0 < seconds:
        frame = move(time.monotonic() - t0)
        mini.set_target(head=frame.head_pose(), antennas=frame.antennas_rad(),
                        body_yaw=frame.body_yaw_rad())
        time.sleep(0.02)


def main() -> int:
    wanted = sys.argv[1:] or list(ALL)
    unknown = [n for n in wanted if n not in ALL]
    if unknown:
        print(f"Unknown move(s): {unknown}. Available: {', '.join(ALL)}")
        return 1

    with ReachyMini() as mini:
        try:
            if len(wanted) == 1:
                play(mini, wanted[0], None)  # loop until Ctrl-C
            else:
                for name in wanted:
                    play(mini, name, 6.0)
        except KeyboardInterrupt:
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
