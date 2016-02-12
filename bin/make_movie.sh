#!/bin/bash

avconv -framerate 2 -i plots/plot_%05d.png -c:v libx264 -r 30 -pix_fmt yuv420p $*

