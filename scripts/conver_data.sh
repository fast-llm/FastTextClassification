python ./src/data/convert_raw_data.py \
    --data_path "./data/ChnSentiCorp_htl_all/input/ChnSentiCorp_htl_all.csv" \
    --save_path "./data/ChnSentiCorp_htl_all/data" \
    --split_ratio "8 1 1" \
    --num_class 2 \
    --label_col label --text_col review