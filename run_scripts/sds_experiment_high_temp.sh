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
	--lr=0.025\
	--lr_temp=0.4 \
	--use_dip=${2} \
	--l2_weight=${3} \
	--semantic_weight=${4} \
	--geometric_weight=${5} \
	--style_weight=${6} \
	--straight_through=${7} \
	--num_colors=${8} \
	--sds_weight=${10} \
	--sds_prompt="$9" \
	--old_method=True \
	--canvas_h=${12} \
	--canvas_w=${13} \
	--no_palette_mode=${11}
}

# STARTING RUN:

# With straight through

input=lily_cropped.jpg
use_dip=False
l2_weight=100.0
sds_weight=1.0
semantic_weight=0.0
geometric_weight=0.0
style_weight=0.0
straight_through=True
num_colors=6
sds_prompt="an image of a smiling woman in 16bit pixelart style"
no_palette_mode=False
canvas_h=32
canvas_w=32

train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
$style_weight $straight_through $num_colors "$sds_prompt" $sds_weight $no_palette_mode \
$canvas_h $canvas_w

input=me3.jpg
use_dip=False
l2_weight=100.0
sds_weight=1.0
semantic_weight=0.0
geometric_weight=0.0
style_weight=0.0
straight_through=True
num_colors=6
sds_prompt="an image of a man smiling in 16bit pixelart style"
no_palette_mode=False
canvas_h=32
canvas_w=32

train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
$style_weight $straight_through $num_colors "$sds_prompt" $sds_weight $no_palette_mode \
$canvas_h $canvas_w

input=biden_cropped.jpg
use_dip=False
l2_weight=100.0
sds_weight=1.0
semantic_weight=0.0
geometric_weight=0.0
style_weight=0.0
straight_through=True
num_colors=6
sds_prompt="an image of joe biden in 16bit pixelart style"
no_palette_mode=False
canvas_h=32
canvas_w=32

train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
$style_weight $straight_through $num_colors "$sds_prompt" $sds_weight $no_palette_mode \
$canvas_h $canvas_w


input=strawberry.jpeg
use_dip=False
l2_weight=100.0
sds_weight=1.0
semantic_weight=0.0
geometric_weight=0.0
style_weight=0.0
straight_through=True
num_colors=6
sds_prompt="an image of a cute baby girl in 8bit pixelart style"
no_palette_mode=False
canvas_h=35
canvas_w=35

train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
$style_weight $straight_through $num_colors "$sds_prompt" $sds_weight $no_palette_mode \
$canvas_h $canvas_w

input=obama.png
use_dip=False
l2_weight=100.0
sds_weight=1.0
semantic_weight=0.0
geometric_weight=0.0
style_weight=0.0
straight_through=True
num_colors=6
sds_prompt="an image of barack obama in 8bit pixelart style"
no_palette_mode=False
canvas_h=32
canvas_w=32

train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
$style_weight $straight_through $num_colors "$sds_prompt" $sds_weight $no_palette_mode \
$canvas_h $canvas_w

input=camel.png
use_dip=False
l2_weight=100.0
sds_weight=1.0
semantic_weight=0.0
geometric_weight=0.0
style_weight=0.0
straight_through=True
num_colors=5
sds_prompt="an image of a cute camel over a white background in 8bit pixelart style"
no_palette_mode=False
canvas_h=32
canvas_w=32

train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
$style_weight $straight_through $num_colors "$sds_prompt" $sds_weight $no_palette_mode \
$canvas_h $canvas_w

input=clown_fish.png
use_dip=False
l2_weight=100.0
sds_weight=1.0
semantic_weight=0.0
geometric_weight=0.0
style_weight=0.0
straight_through=True
num_colors=5
sds_prompt="an image of a cute clown fish over a white background in 8bit pixelart style"
no_palette_mode=False
canvas_h=32
canvas_w=32

train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
$style_weight $straight_through $num_colors "$sds_prompt" $sds_weight $no_palette_mode \
$canvas_h $canvas_w

input=flamingo.png
use_dip=False
l2_weight=100.0
sds_weight=1.0
semantic_weight=0.0
geometric_weight=0.0
style_weight=0.0
straight_through=True
num_colors=5
sds_prompt="an image of a cute flamingo over a white background in 8bit pixelart style"
no_palette_mode=False
canvas_h=32
canvas_w=32

train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
$style_weight $straight_through $num_colors "$sds_prompt" $sds_weight $no_palette_mode \
$canvas_h $canvas_w