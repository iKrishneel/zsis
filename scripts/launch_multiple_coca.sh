#!/usr/bin/env bash

tmux new-session -d -s coca

# split window vertically to panels 0 and 1
tmux split-window -v 

# split panel 1 horizontally to panels 1 and 2 (left to right)
tmux split-window -v -t 0
tmux split-window -v -t 1

tmux send-keys -t 0 "python coca_caption_gen.py --start 0 --end 29572 --device cuda:0" C-m
tmux send-keys -t 1 "python coca_caption_gen.py --start 29572 --end 59142 --device cuda:1" C-m
tmux send-keys -t 2 "python coca_caption_gen.py --start 59142 --end 88713 --device cuda:2" C-m
tmux send-keys -t 3 "python coca_caption_gen.py --start 88713 --end 118284 --device cuda:3" C-m

tmux attach-session -d -t coca
