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
	-o results/${1}/h_${12}_w_${13}_c_${8}_lr_${15}_lrp_${14}_l2_${3}/ \
	--sds_control=True \
	--lr_palette=${14} \
	--lr=${15} \
	--lr_warmup=0.009 \
	--epochs=5000 \
	--lr_temp=0.001 \
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

input=me_nobg.png
use_dip=False
l2_weight=10.0
sds_weight=1.0
semantic_weight=0.0
geometric_weight=0.0
style_weight=0.0
straight_through=True
num_colors=10
sds_prompt="an image of a man in 16bit pixelart style"
no_palette_mode=False
canvas_h=32
canvas_w=32
palette_lr=0.00001
lr=0.0023

train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
$style_weight $straight_through $num_colors "$sds_prompt" $sds_weight $no_palette_mode \
$canvas_h $canvas_w $palette_lr $lr


input=lily_nobg.png
use_dip=False
l2_weight=10.0
sds_weight=1.0
semantic_weight=0.0
geometric_weight=0.0
style_weight=0.0
straight_through=True
num_colors=12
sds_prompt="an image of a smiling woman over a white background in 16bit pixelart style"
no_palette_mode=False
canvas_h=32
canvas_w=32
palette_lr=0.00001
lr=0.0023

train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
$style_weight $straight_through $num_colors "$sds_prompt" $sds_weight $no_palette_mode \
$canvas_h $canvas_w $palette_lr $lr

input=rock_nobg.png
use_dip=False
l2_weight=10.0
sds_weight=1.0
semantic_weight=0.0
geometric_weight=0.0
style_weight=0.0
straight_through=True
num_colors=12
sds_prompt="an image of dwayne the rock johnson in 16bit pixelart style"
no_palette_mode=False
canvas_h=32
canvas_w=32
palette_lr=0.00001
lr=0.008

train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
$style_weight $straight_through $num_colors "$sds_prompt" $sds_weight $no_palette_mode \
$canvas_h $canvas_w $palette_lr $lr


input=obama_nobg.png
use_dip=False
l2_weight=10.0
sds_weight=1.0
semantic_weight=0.0
geometric_weight=0.0
style_weight=0.0
straight_through=True
num_colors=10
sds_prompt="an image of barack obama in 16bit pixelart style"
no_palette_mode=False
canvas_h=32
canvas_w=32
palette_lr=0.00001
lr=0.008

train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
$style_weight $straight_through $num_colors "$sds_prompt" $sds_weight $no_palette_mode \
$canvas_h $canvas_w $palette_lr $lr

#input=me_nobg.png
#use_dip=False
#l2_weight=20000.0
#sds_weight=1.0
#semantic_weight=0.0
#geometric_weight=0.0
#style_weight=0.0
#straight_through=True
#num_colors=10
#sds_prompt="an image of a smiling man in 16bit pixelart style"
#no_palette_mode=False
#canvas_h=32
#canvas_w=32
#palette_lr=0.0001
#lr=0.0023
#
#train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
#$style_weight $straight_through $num_colors "$sds_prompt" $sds_weight $no_palette_mode \
#$canvas_h $canvas_w $palette_lr $lr
#
#input=obama_nobg.png
#use_dip=False
#l2_weight=20000.0
#sds_weight=1.0
#semantic_weight=0.0
#geometric_weight=0.0
#style_weight=0.0
#straight_through=True
#num_colors=12
#sds_prompt="an image of barack obama in 16bit pixelart style"
#no_palette_mode=False
#canvas_h=32
#canvas_w=32
#palette_lr=0.0001
#lr=0.0023
#
#train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
#$style_weight $straight_through $num_colors "$sds_prompt" $sds_weight $no_palette_mode \
#$canvas_h $canvas_w $palette_lr $lr
#
#input=lily_nobg.png
#use_dip=False
#l2_weight=20000.0
#sds_weight=1.0
#semantic_weight=0.0
#geometric_weight=0.0
#style_weight=0.0
#straight_through=True
#num_colors=12
#sds_prompt="an image of a smiling woman over a white background in 16bit pixelart style"
#no_palette_mode=False
#canvas_h=32
#canvas_w=32
#palette_lr=0.0001
#lr=0.0023
#
#train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
#$style_weight $straight_through $num_colors "$sds_prompt" $sds_weight $no_palette_mode \
#$canvas_h $canvas_w $palette_lr $lr
#
#input=me_nobg.png
#use_dip=False
#l2_weight=1000.0
#sds_weight=1.0
#semantic_weight=0.0
#geometric_weight=0.0
#style_weight=0.0
#straight_through=True
#num_colors=10
#sds_prompt="an image of a smiling man in 16bit pixelart style"
#no_palette_mode=False
#canvas_h=32
#canvas_w=32
#palette_lr=0.00001
#lr=0.004
#
#train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
#$style_weight $straight_through $num_colors "$sds_prompt" $sds_weight $no_palette_mode \
#$canvas_h $canvas_w $palette_lr $lr
#
#input=obama_nobg.png
#use_dip=False
#l2_weight=1000.0
#sds_weight=1.0
#semantic_weight=0.0
#geometric_weight=0.0
#style_weight=0.0
#straight_through=True
#num_colors=10
#sds_prompt="an image of barack obama in 16bit pixelart style"
#no_palette_mode=False
#canvas_h=32
#canvas_w=32
#palette_lr=0.00001
#lr=0.004
#
#train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
#$style_weight $straight_through $num_colors "$sds_prompt" $sds_weight $no_palette_mode \
#$canvas_h $canvas_w $palette_lr $lr
#
#input=lily_nobg.png
#use_dip=False
#l2_weight=1000.0
#sds_weight=1.0
#semantic_weight=0.0
#geometric_weight=0.0
#style_weight=0.0
#straight_through=True
#num_colors=10
#sds_prompt="an image of a smiling woman over a white background in 16bit pixelart style"
#no_palette_mode=False
#canvas_h=32
#canvas_w=32
#palette_lr=0.00001
#lr=0.004
#
#train $input $use_dip $l2_weight $semantic_weight $geometric_weight \
#$style_weight $straight_through $num_colors "$sds_prompt" $sds_weight $no_palette_mode \
#$canvas_h $canvas_w $palette_lr $lr