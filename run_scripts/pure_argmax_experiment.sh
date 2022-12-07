#!/bin/bash
echo "Starting Run!"

# Reading arguments:
gpu_num=0
while getopts g: flag
do
    case "${flag}" in
        g) gpu_num=${OPTARG};;
    esac
done

# Setting GPU:
echo "Running on GPU: $gpu_num";
export CUDA_VISIBLE_DEVICES=$gpu_num

# Rendering function template:
train() {
	# Train:
	echo "Starting Training..."
	python pixelArtClipasso.py \
	-i ${1}/ \
	--canvas_h=16 \
	--canvas_w=16 \
	--use_dip=${2} \
	--l2_weight=${3} \
	--semantic_weight=${4} \
	--geometric_weight=${5} \
	--style_weight=${6} \
	--straight_through=${7} \
	--num_colors=${8} \
	--by_distance=${9} \
	--old_method=${10}
}

# STARTING RUN:

# With pure argmax

input=bowser.png
use_dip=True
l2_weight=1.0
semantic_weight=0.0
geometric_weight=0.0
style_weight=0.0
straight_through=True
num_colors=4
by_distance=False
old_method=False

train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
$style_weight $straight_through $num_colors $by_distance $old_method

input=camel.png
use_dip=True
l2_weight=1.0
semantic_weight=0.0
geometric_weight=0.0
style_weight=0.0
straight_through=True
num_colors=4
by_distance=False
old_method=False

train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
$style_weight $straight_through $num_colors $by_distance $old_method

input=clown_fish.png
use_dip=True
l2_weight=1.0
semantic_weight=0.0
geometric_weight=0.0
style_weight=0.0
straight_through=True
num_colors=4
by_distance=False
old_method=False

train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
$style_weight $straight_through $num_colors $by_distance $old_method

input=obama.png
use_dip=True
l2_weight=1.0
semantic_weight=0.0
geometric_weight=0.0
style_weight=0.0
straight_through=True
num_colors=4
by_distance=False
old_method=False

train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
$style_weight $straight_through $num_colors $by_distance $old_method

input=pixil_mario_wbg.png
use_dip=True
l2_weight=1.0
semantic_weight=0.0
geometric_weight=0.0
style_weight=0.0
straight_through=True
num_colors=4
by_distance=False
old_method=False

train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
$style_weight $straight_through $num_colors $by_distance $old_method

# Without pure argmax

input=bowser.png
use_dip=True
l2_weight=1.0
semantic_weight=0.0
geometric_weight=0.0
style_weight=0.0
straight_through=True
num_colors=4
by_distance=False
old_method=True

train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
$style_weight $straight_through $num_colors $by_distance $old_method

input=camel.png
use_dip=True
l2_weight=1.0
semantic_weight=0.0
geometric_weight=0.0
style_weight=0.0
straight_through=True
num_colors=4
by_distance=False
old_method=True

train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
$style_weight $straight_through $num_colors $by_distance $old_method

input=clown_fish.png
use_dip=True
l2_weight=1.0
semantic_weight=0.0
geometric_weight=0.0
style_weight=0.0
straight_through=True
num_colors=4
by_distance=False
old_method=True

train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
$style_weight $straight_through $num_colors $by_distance $old_method

input=obama.png
use_dip=True
l2_weight=1.0
semantic_weight=0.0
geometric_weight=0.0
style_weight=0.0
straight_through=True
num_colors=4
by_distance=False
old_method=True

train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
$style_weight $straight_through $num_colors $by_distance $old_method

input=pixil_mario_wbg.png
use_dip=True
l2_weight=1.0
semantic_weight=0.0
geometric_weight=0.0
style_weight=0.0
straight_through=True
num_colors=4
by_distance=False
old_method=True

train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
$style_weight $straight_through $num_colors $by_distance $old_method