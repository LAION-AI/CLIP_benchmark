#!/bin/bash

rsync -avz \
  --exclude .git/ \
   ../CLIP_benchmark spratt3@klone.hyak.uw.edu:/gscratch/raivn/spratt
