cfg_files=("concat_cfg1.yaml" "concat_cfg2.yaml")

for cfg_file in "${cfg_files[@]}"; do
    echo "Training $cfg_file"
    python3 concat_train.py "$cfg_file"
done

