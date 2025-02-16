#!/bin/sh
export APPTAINER_CACHEDIR=/sda2/
export APPTAINER_TMPDIR=/sda2/

apptainer build --fakeroot apptainer.sif apptainer.def