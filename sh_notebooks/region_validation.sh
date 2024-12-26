name=train_west_bengal_val_punjab
task=obb
suffix=v1
image_size=640
batch_size=128
epochs=300
device=1
train_model=yolo11m-obb.pt
model_dir=/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/runs/obb/$name\_$task\_$suffix\_$train_model\_$image_size\_$batch_size\_$epochs
model=best.pt
base_path=/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns
data_path=$base_path/yaml_data_dir/val.yaml
conf=0.25
iou_thres=0.50
save_json=True
experiment_name=$name\_$task\_$suffix\_$model\_$image_size\_$batch_size\_$epochs
log_file=$base_path/region_performance_logs/$experiment_name

echo "Name: $name"
echo "Task: $task"
echo "Suffix: $suffix"
echo "Model: $model"
echo "Image Size: $image_size"
echo "Batch Size: $batch_size"
echo "Epochs: $epochs"
echo "Device: $device"
echo "Base Path: $base_path"
echo "Data Path: $data_path"
echo "Val: $val"
echo "Conf: $conf"
echo "Save Txt: $save_txt"
echo "Log File: $log_file"
echo "Experiment Name: $experiment_name"

nohup yolo obb val model=$model_dir/$model\
    data=$data_path\
    imgsz=$image_size\
    device=$device\
    batch=$batch_size\
    conf=$conf\
    save_txt=$save_txt\
    name=$base_path/runs/obb/$experiment_name\
    save=True\
    > $log_file.log 2>&1 &

echo "Started validation for $experiment_name"    
    

