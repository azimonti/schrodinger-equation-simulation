#!/bin/bash

# Input / Output data
VIDEO1="VIDEO_PROBABILITY_PLOT"
VIDEO2="VIDEO_WAVEPACKET"
OUTPUT="VIDEO_OUTPUT"

# Set variables for scale percentage, border size, and margins
SCALE_PERCENTAGE=0.27  # 27% scaling
BORDER_SIZE=5          # Border size in pixels
MARGIN_X=0.025         # Margin as a percentage of the width
MARGIN_Y=0.05          # Margin as a percentage of the height

ffmpeg -i "${VIDEO1}".mp4 -i "${VIDEO2}".mp4 -filter_complex "
    [1:v]scale=iw*${SCALE_PERCENTAGE}:ih*${SCALE_PERCENTAGE}[scaled];
    [scaled]pad=iw+$((${BORDER_SIZE}*2)):ih+$((${BORDER_SIZE}*2)):${BORDER_SIZE}:${BORDER_SIZE}:white[bordered];
    [0:v][bordered]overlay=W*${MARGIN_X}:H*${MARGIN_Y}
" -c:a copy "${OUTPUT}".mp4
