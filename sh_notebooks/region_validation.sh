base_state="west_bengal"
target_state="bihar"
test_state="haryana"
ratio=0.25
name="train_${base_state}_val_${target_state}_${ratio}"
task=obb
suffix=v1
image_size=640
batch_size=128
epochs=300
device=2
train_model=yolo11m-obb.pt
train_test_zone=train_${base_state}_test_${test_state}_${ratio}
model_dir=/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/runs/obb/$train_test_zone\_$task\_$suffix\_$train_model\_$image_size\_$batch_size\_$epochs/weights
model=best.pt
base_path=/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns
data_path=$base_path/yaml_data_dir/val.yaml
conf=0.25
iou_thres=0.50
save_json=True
experiment_name=$name\_$task\_$suffix\_$model\_$image_size\_$batch_size\_$epochs\_$conf\_$iou_thres
log_file=$base_path/region_performance_logs/$experiment_name

echo "Name: $name"
echo "ratio: $ratio"
echo "Task: $task"
echo "Suffix: $suffix"
echo "Model: $model"
echo "Image Size: $image_size"
echo "Batch Size: $batch_size"
echo "Epochs: $epochs"
echo "Device: $device"
echo "Base Path: $base_path"
echo "Data Path: $data_path"
echo "Conf: $conf"
echo "Save json: $save_json"
echo "Log File: $log_file"
echo "Experiment Name: $experiment_name"

nohup yolo obb val model=$model_dir/$model\
    data=$data_path\
    imgsz=$image_size\
    device=$device\
    batch=$batch_size\
    conf=$conf\
    iou=$iou_thres\
    save_json=$save_json\
    name=$base_path/runs/obb/$experiment_name\
    save=True\
    > $log_file.log 2>&1 &

echo "Started validation for $experiment_name"    
    

