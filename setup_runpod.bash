#!/usr/bin/env bash

# This script is used to setup the runpod environment


SSH_PORT=16504
SSH_HOST=209.170.80.132
SSH_USER=root
WORKDIR=/workspace/mml

#scp -i ~/.ssh/id_ed25519 -P $SSH_PORT -r ./* $SSH_USER@$SSH_HOST:$WORKDIR

scp -i ~/.ssh/id_ed25519 -P $SSH_PORT -r ~/.ssh/id_ed25519 $SSH_USER@$SSH_HOST:~/.ssh/id_ed25519
scp -i ~/.ssh/id_ed25519 -P $SSH_PORT -r ~/.ssh/id_ed25519.pub $SSH_USER@$SSH_HOST:~/.ssh/id_ed25519.pub

# execute remotely
ssh RunPodMML << ENDSSH

chmod 600 ~/.ssh/id_25519

apt update && apt install tmux neovim less tree unzip htop lshw ffmpeg nvidia-cuda-toolkit libfuse2 git nano -y
mv /root/.cache /workspace/.cache && ln -s /workspace/.cache /root/.cache

apt update -y
apt upgrade -y
apt dist-upgrade -y
apt autoremove -y
apt autoclean -y

mkdir -p $WORKDIR
cd $WORKDIR

python3.11 -m venv env
source env/bin/activate
pip install --upgrade pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# tmux config
cat << EOF >| ~/.tmux.conf
unbind C-b
set-option -g prefix \`
bind-key \` send-prefix

# split panes using | and -
bind | split-window -h
bind - split-window -v
unbind '"'
unbind %

# switch panes using Alt-arrow without prefix
bind -n M-Left select-pane -L
bind -n M-Right select-pane -R
bind -n M-Up select-pane -U
bind -n M-Down select-pane -D

set -g mouse on

set-option -g history-limit 50000

set -g default-terminal 'screen-256color'
set -sa terminal-features ',xterm-256color:RGB'
EOF

cat << EOF >> ~/.bashrc
# Avoid duplicates
HISTCONTROL=ignoredups:erasedups
# When the shell exits, append to the history file instead of overwriting it

set -o noclobber
shopt -s checkwinsize
shopt -s no_empty_cmd_completion
shopt -s histappend
shopt -s autocd
shopt -s dirspell
shopt -s cdspell
shopt -s cmdhist
shopt -s globstar
shopt -s extglob

tmux_run() {
parent_process=$(ps -p $PPID -o comm=)
if [[ "$parent_process" != "code" ]]; then
    if [ -n "$PS1" ] && [ -z "$TMUX" ]; then
    # Adapted from https://unix.stackexchange.com/a/176885/347104
    # Create session 'main' or attach to 'main' if already exists.
    tmux new-session -A -s main
    fi
fi
}
tmux_run
EOF

ENDSSH
