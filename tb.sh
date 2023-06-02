#!/bin/bash

tmux new -d -s tb
tmux send-keys -t tb 'conda activate pt2' C-m
tmux send-keys -t tb 'tensorboard --logdir=outputs/lightning_logs --bind_all' C-m
