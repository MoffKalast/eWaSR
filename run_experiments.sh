#!/bin/bash
cd "$(dirname "$0")"
mkdir -p output

COMMON="--train_config configs/lars_train.yaml --val_config configs/lars_val.yaml --model ewasr_resnet34 --validation --batch_size 12 --gpus 2 --workers 2 --prefetch_factor 1 --sim_interp bilinear --random_seed 0"
INTERVAL=60

names=()
pids=()

run() {
	local name=$1 gpus=$2 port=$3
	shift 3
	echo "starting $name on GPUs $gpus, log in output/$name.log"
	CUDA_VISIBLE_DEVICES=$gpus MASTER_PORT=$port NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 nohup python train.py $COMMON --model_name "$name" "$@" > "output/$name.log" 2>&1 &
	names+=("$name")
	pids+=($!)
}

any_running() {
	for pid in "${pids[@]}"; do
		if kill -0 "$pid" 2>/dev/null; then
			return 0
		fi
	done
	return 1
}

report() {
	echo "--- $(date '+%H:%M:%S')"
	for i in "${!names[@]}"; do
		local log="output/${names[$i]}.log"
		if kill -0 "${pids[$i]}" 2>/dev/null; then
			local line=$(grep '^epoch ' "$log" | tail -n 1)
			echo "${names[$i]}: ${line:-starting up}"
		else
			echo "${names[$i]}: exited, last line: $(tail -n 1 "$log")"
		fi
	done
}

run r2_noise0.067-0.147_e80 0,1 29501 --learning_rate 4e-6 --epochs 80 --noise_std 0.067 0.147
run r2_noise0.133-0.293_e80 2,3 29502 --learning_rate 4e-6 --epochs 80 --noise_std 0.133 0.293
run r2_noise0.2-0.44_e80 4,5 29503 --learning_rate 4e-6 --epochs 80 --noise_std 0.2 0.44
run r2_noise0.25-0.55_e80 6,7 29504 --learning_rate 4e-6 --epochs 80 --noise_std 0.25 0.55

while any_running; do
	sleep $INTERVAL
	report
done

wait
