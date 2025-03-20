#!/usr/bin/env bash
trap "echo '中断信号收到，正在终止所有任务...'; kill 0; exit 1" SIGINT

# echo "=== Step 1: Running data_preprocess.py ==="
# python data_preprocess.py

echo "=== Step 2: Running tool_detection.py and feature_detection.py in parallel ==="
python tool_detection.py &
pid1=$!
python feature_detection.py &
pid2=$!
wait $pid1 $pid2

echo "=== Step 3: Running generate_tool.py and extracate_features.py in parallel ==="
python generate_tool.py &
pid3=$!
python extracate_features.py &
pid4=$!
wait $pid3 $pid4

echo "=== Step 4: Running sbs_detection.py ==="
python sbs_detection.py

echo "=== Step 5: Running sbs_generate_state.py ==="
python sbs_generate_state.py

echo "=== Step 6: Running robust_learning.py, CE_learning.py, GCE_learning.py, and feature_learning.py in parallel ==="
python robust_learning.py &
pid5=$!
python CE_learning.py &
pid6=$!
python GCE_learning.py &
pid7=$!
python feature_learning.py &
pid8=$!
wait $pid5 $pid6 $pid7 $pid8

echo "Pipeline complete."
