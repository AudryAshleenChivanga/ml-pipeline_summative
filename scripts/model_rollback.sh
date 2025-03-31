#!/bin/bash

# Rollback to previous model version
PREV_VERSION=$(ls models/ | grep glaucoma_model | sort -r | sed -n '2p')
ln -sf $PREV_VERSION models/glaucoma_model_latest.h5
echo "Rolled back to $PREV_VERSION"