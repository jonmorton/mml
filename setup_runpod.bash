#!/usr/bin/env bash

# This script is used to setup the runpod environment


SSH_PORT=${SSH_PORT:-18474}
SSH_HOST=${SSH_HOST:-205.196.17.170}
SSH_USER=${SSH_USER:-root}
WORKDIR=${WORKDIR:-/root/mml}

ssh -i ~/.ssh/id_ed25519 -p $SSH_PORT $SSH_USER@$SSH_HOST "apt update -y && apt install -y rsync"

rsync -avrz -e "ssh -i ~/.ssh/id_ed25519 -p ${SSH_PORT}" --progress --exclude "env" --exclude "env2" --exclude ".git" --exclude "models" --exclude "data" --exclude "runs" ./ $SSH_USER@$SSH_HOST:$WORKDIR

scp -i ~/.ssh/id_ed25519 -P $SSH_PORT ~/.ssh/id_ed25519 $SSH_USER@$SSH_HOST:~/.ssh/id_ed25519
scp -i ~/.ssh/id_ed25519 -P $SSH_PORT ~/.ssh/id_ed25519.pub $SSH_USER@$SSH_HOST:~/.ssh/id_ed25519.pub

# execute remotely
ssh RunPodMML << ENDSSH

cp $WORKDIR/tmux.conf ~/.tmux.conf

git config --global user.email "jon@jmorton.com"
git config --global user.name "Jon Morton"

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
pip install torch -U torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -U wandb unsloth tensorboard vllm zstandard polars stable-baselines3 transformers

cat << 'EOF' >> ~/.bashrc
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
  if [[ "" != "code" ]]; then
   if [ -z "$TMUX" ]; then
      # Adapted from https://unix.stackexchange.com/a/176885/347104
      # Create session 'main' or attach to 'main' if already exists.
      tmux new-session -A -s main
    fi
  fi
}

tmux_run
EOF

ENDSSH
