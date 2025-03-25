cfg_files=("concat_cfg1.yaml" "concat_cfg2.yaml" "concat_cfg3.yaml" "concat_cfg4.yaml" "concat_cfg5.yaml")
# cfg_files=("concat_cfg2.yaml")

for cfg_file in "${cfg_files[@]}"; do
    echo "Training $cfg_file"
    python3 concat_train.py "$cfg_file"
done

