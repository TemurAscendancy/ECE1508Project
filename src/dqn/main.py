
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
    p.add_argument("--data", default="~/vgc_endgames_ma")
    p.add_argument("--episodes", type=int, help="num_episodes (train) or num_test_episodes (play)")
    p.add_argument("--max_records", type=int)
    p.add_argument("--learning_starts", type=int)
    p.add_argument("--run_dir")
    args = p.parse_args()

    hyperparameters = Hyperparameters(data_dir=args.data)
    # hyperparameters.change(epsilon_decay=.999)
    if args.episodes is not None:
        hyperparameters.change(**{("num_test_episodes" if args.play else "num_episodes"): args.episodes})
    if args.max_records is not None:
        hyperparameters.change(max_records=args.max_records)
    if args.learning_starts is not None:
        hyperparameters.change(learning_starts=args.learning_starts)
    if args.run_dir is not None:
        hyperparameters.change(run_dir=os.path.abspath(args.run_dir))

    train = not args.play

    # Run
    DRL = DQL(hyperparameters, train_mode=train) # Define the instance
    # Train
    if train:
        DRL.train()
    else:
        DRL.play()
