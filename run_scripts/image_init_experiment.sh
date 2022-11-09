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
	--use_dip=${2} \
	--l2_weight=${3} \
	--semantic_weight=${4} \
	--geometric_weight=${5} \
	--style_weight=${6} \
	--straight_through=${7} \
	--num_colors=${8} \
	--init_image=${9}
}

# STARTING RUN:

# With straight through

input=flamingo.png
use_dip=True
l2_weight=1.0
semantic_weight=0.0
geometric_weight=0.0
style_weight=0.0
straight_through=True
num_colors=6
init_image=False

train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
$style_weight $straight_through $num_colors $init_image

input=3dMario.png
use_dip=True
l2_weight=1.0
semantic_weight=0.0
geometric_weight=0.0
style_weight=0.0
straight_through=True
num_colors=6
init_image=False

train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
$style_weight $straight_through $num_colors $init_image

input=bowser.png
use_dip=True
l2_weight=1.0
semantic_weight=0.0
geometric_weight=0.0
style_weight=0.0
straight_through=True
num_colors=6
init_image=False

train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
$style_weight $straight_through $num_colors $init_image

input=camel.png
use_dip=True
l2_weight=1.0
semantic_weight=0.0
geometric_weight=0.0
style_weight=0.0
straight_through=True
num_colors=6
init_image=False

train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
$style_weight $straight_through $num_colors $init_image

input=clown_fish.png
use_dip=True
l2_weight=1.0
semantic_weight=0.0
geometric_weight=0.0
style_weight=0.0
straight_through=True
num_colors=6
init_image=False

train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
$style_weight $straight_through $num_colors $init_image

input=goldfish.png
use_dip=True
l2_weight=1.0
semantic_weight=0.0
geometric_weight=0.0
style_weight=0.0
straight_through=True
num_colors=6
init_image=False

train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
$style_weight $straight_through $num_colors $init_image

input=obama.png
use_dip=True
l2_weight=1.0
semantic_weight=0.0
geometric_weight=0.0
style_weight=0.0
straight_through=True
num_colors=6
init_image=False

train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
$style_weight $straight_through $num_colors $init_image