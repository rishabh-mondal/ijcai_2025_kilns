#!/bin/bash

train="trench_width_delhi_ncr"
task="obb"
suffix="v2"
root_path="/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns"
base_path="/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/gms_data"
state_part_name="delhi_ncr_data"
data="$base_path/$state_part_name/images_main"
imgsz=640
epochs=100
device=1
model="/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/runs/aa/train_trench_width_delhi_ncr_test_delhi_ncr_bricks_92_obb_v1_yolo11m-obb.pt_640_128_100/weights/best.pt"
log_dir="$root_path/region_performance_logs"
log_file="$log_dir/$state_part_name.log"

# Creating experiment name
experimentName="${train}_${task}_${suffix}_${imgsz}_${epochs}"

# Print out the variables
echo "Train:" $train
echo "Task:" $task
echo "Suffix:" $suffix
echo "Root Path:" $root_path
echo "Base Path:" $base_path
echo "Data:" $data
echo "Image Size:" $imgsz
echo "Epochs:" $epochs
echo "Device:" $device
echo "Experiment Name:" $experimentName
echo "Model:" $model
echo "Log Dir:" $log_dir
echo "Log File:" $log_file

# Execute the YOLO prediction command with nohup
nohup yolo obb predict model=$model source=$data conf=0.25 iou=0.5 imgsz=$imgsz device=$device name="$root_path/predict/$experimentName/$state_part_name" save_txt=True save=False save_conf=True save_crop=False verbose=True > $log_file 2>&1 &

echo "Job fired!"
