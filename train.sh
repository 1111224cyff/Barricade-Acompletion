export MASTER_ADDR="localhost"
export MASTER_PORT=12348
export RANK=0
export WORLD_SIZE=1

source /D/anaconda/bin/activate
conda activate acomp_barricade

num_proc=2

dataset=COCOA # change the dataset name here COCOA or KINS

work_path=experiments/$dataset/pcnet_m

OMP_NUM_THREADS=1 python main.py --config $work_path/config_train.yaml --launcher pytorch --exp_path experiments/$dataset/pcnet_m_barricade

