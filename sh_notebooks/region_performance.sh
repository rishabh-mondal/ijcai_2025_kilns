train_state="haryana"
test_state="bihar"
name="train_${train_state}_test_${test_state}"
ratio=same_class_count
task=obb
suffix=v1
model_dir=/home/patel_zeel/kiln_compass_24
model=yolo11m-obb.pt
image_size=640
batch_size=128
epochs=300
device=0
base_path=/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns
data_path=$base_path/yaml_data_dir/train_test.yaml
val=False
# val_interval=10 #not supported in train mode 
save_conf=True
save_txt=False
experiment_name=$name\_$ratio\_$task\_$suffix\_$model\_$image_size\_$batch_size\_$epochs
log_file=$base_path/region_performance_logs/$experiment_name

echo "Name: $name"
echo "Task: $task"
echo "Ratio: $ratio"
echo "Suffix: $suffix"
echo "Model: $model"
echo "Image Size: $image_size"
echo "Batch Size: $batch_size"
echo "Epochs: $epochs"
echo "Device: $device"
echo "Base Path: $base_path"
echo "Data Path: $data_path"
echo "Val: $val"
echo "Save Conf: $save_conf"
echo "Save Txt: $save_txt"
echo "Log File: $log_file"
echo "Experiment Name: $experiment_name"

nohup yolo obb train model=$model_dir/$model\
    data=$data_path\
    imgsz=$image_size\
    epochs=$epochs\
    device=$device\
    val=$val\
    batch=$batch_size\
    save_conf=$save_conf\
    save_txt=$save_txt\
    name=$base_path/runs/obb/$experiment_name\
    save=True\
    > $log_file.log 2>&1 &

echo "Started training for $experiment_name"    
    

