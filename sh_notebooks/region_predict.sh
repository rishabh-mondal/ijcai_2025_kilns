#!/bin/bash
train="train_jcss_processed_data_test_bihar_same_class_count_10_120_1000"
task="obb"
suffix="v2"
root_path="/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns"
state_part_name="mymensingh"
base_path="../data/processed_data/$state_part_name"
data="$base_path/images"
imgsz=640
epochs=300
device=1
model="../runs/obb/train_jcss_processed_data_test_bihar_same_class_count_10_120_1000_obb_v1_640_64_300/weights/best.pt"
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
nohup yolo obb predict model=$model source=$data conf=0.25 iou=0.5 imgsz=$imgsz device=$device name="$root_path/data/predict/processed_labels/$state_part_name" save_txt=True save=False save_conf=True save_crop=False verbose=True > $log_file 2>&1 &
echo "Job fired!"
