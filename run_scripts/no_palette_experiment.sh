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
	--canvas_h=224 \
	--canvas_w=224 \
	--use_dip=${2} \
	--l2_weight=${3} \
	--semantic_weight=${4} \
	--geometric_weight=${5} \
	--style_weight=${6} \
	--straight_through=${7} \
	--num_colors=${8} \
	--no_palette_mode=${9} \
	--style_weight=${10} \
	--shift_aware_weight=${11}
}

# STARTING RUN:

# With straight through
input=flamingo.png
use_dip=True
l2_weight=0.0
semantic_weight=1.0
geometric_weight=0.01
style_weight=0.0
straight_through=True
num_colors=5
no_palette_mode=True
style_weight=0.0
shift_aware_weight=0.0
style_prompt="none"

train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
$style_weight $straight_through $num_colors $no_palette_mode $style_weight  \
$shift_aware_weight $style_prompt

input=camel.png
use_dip=True
l2_weight=0.0
semantic_weight=1.0
geometric_weight=0.01
style_weight=0.0
straight_through=True
num_colors=5
no_palette_mode=True
style_weight=0.0
shift_aware_weight=0.0
style_prompt="none"

train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
$style_weight $straight_through $num_colors $no_palette_mode $style_weight  \
$shift_aware_weight $style_prompt

input=goldfish.png
use_dip=True
l2_weight=0.0
semantic_weight=1.0
geometric_weight=0.01
style_weight=0.0
straight_through=True
num_colors=5
no_palette_mode=True
style_weight=0.0
shift_aware_weight=0.0
style_prompt="none"

train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
$style_weight $straight_through $num_colors $no_palette_mode $style_weight  \
$shift_aware_weight $style_prompt