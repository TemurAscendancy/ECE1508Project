
# DQN on 2v2 VGC endgames. template Code is borrowed from assignment 3
# train w/ python src/dqn/main.py
# eval: python src/dqn/main.py --play

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse

from Hyperparameters import Hyperparameters
from DQL import DQL

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--play", action="store_true", help="evaluate saved weights instead of training")
    p.add_argument("--weights", nargs="+")
    p.add_argument("--data", default="~/vgc_endgames_ma")
    p.add_argument("--episodes", type=int, help="num_episodes (train) or num_test_episodes (play)")
    p.add_argument("--max_records", type=int)
    p.add_argument("--learning_starts", type=int)
    p.add_argument("--run_dir")
    p.add_argument("--hp_shaping", type=float,
                   help="lambda for the HP-difference shaping reward (0 = sparse +-1)")
    p.add_argument("--double_dqn", action="store_true", help="use the Double DQN target")
    p.add_argument("--per_mon", action="store_true",
                   help="one Q head per pokemon (2 x 25 actions) instead of the 625 joint actions")
    args = p.parse_args()

    hyperparameters = Hyperparameters(data_dir=args.data)
    # hyperparameters.change(epsilon_decay=.999)
    if args.episodes is not None:
        hyperparameters.change(**{("num_test_episodes" if args.play else "num_episodes"): args.episodes})
    if args.max_records is not None:
        hyperparameters.change(max_records=args.max_records)
    if args.learning_starts is not None:
        hyperparameters.change(learning_starts=args.learning_starts)
    if args.hp_shaping is not None:
        hyperparameters.change(hp_shaping=args.hp_shaping)
    if args.double_dqn:
        hyperparameters.change(double_dqn=True)
    if args.per_mon:
        hyperparameters.change(per_mon=True)
    if args.run_dir is not None:
        hyperparameters.change(run_dir=os.path.abspath(args.run_dir))

    train = not args.play

    # Run
    DRL = DQL(hyperparameters, train_mode=train) # Define the instance
    # Train
    if train:
        DRL.train()
    else:
        DRL.play(args.weights)
