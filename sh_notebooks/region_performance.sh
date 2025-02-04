train_state="m0_obb_without_empty_train_swinir_4x"
test_state="m0_obb_without_empty_val_swinir_4x"
name="train_${train_state}_test_${test_state}"
ratio=80_20
task=obb
suffix=v2
model_dir="/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/runs/obb_m0/train_m0_obb_without_empty_train_swinir_4x_test_m0_obb_without_empty_val_swinir_4x_80_20_obb_v1_2560_8_100/weights"
model=$model_dir/best.pt
image_size=2560
batch_size=8
# save_period=100
epochs=100
device=3
base_path=/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns
data_path=$base_path/yaml_data_dir/train_test.yaml
val=False
# val_interval=10 #not supported in train mode 
save_conf=True
save_txt=False
experiment_name=$name\_$ratio\_$task\_$suffix\_$image_size\_$batch_size\_$epochs
log_file=$base_path/m0_logs/$experiment_name

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
echo "Save Period: $save_period"
echo "Log File: $log_file"
echo "Experiment Name: $experiment_name"

nohup yolo obb train model=$model\
    data=$data_path\
    imgsz=$image_size\
    epochs=$epochs\
    device=$device\
    val=$val\
    workers=8\
    batch=$batch_size\
    save_conf=$save_conf\
    save_txt=$save_txt\
    exist_ok=True\
    name=$base_path/runs/obb_m0/$experiment_name\
    save=True\
    > $log_file.log 2>&1 &

echo "Started training for $experiment_name"    
    

