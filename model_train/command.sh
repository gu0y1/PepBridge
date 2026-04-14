##train
nohup python pepbridge_train_5folds.py \
     data_path=path_to_data_fold  \
    path=path_to_save/checkpoints_multi_lora_align3_ln/ \
    pep_align=True all_align=-3 ln=True > checkpoints_multi_lora_align3_ln.out 2>&1 &

##test
nohup python pepbridge_test_5folds.py \
     data_path=path_to_data_fold  \
    path=path_to_PepBridge/doc/checkpoints_multi_lora_align3_ln \
    use_lora=True > log/pepbridge_test_5folds_align3_ln.out 2>&1 &

##finetune
nohup python finetune_train.py \
    csv_path=path_to_PepBridge/data/B0801_RAKFKQLL_EBV.csv \
    save_path=path_to_PepBridge/data/checkpoints_finetune/EBV/avaj \
    load_path=path_to_PepBridge/doc/checkpoints_multi_lora_align3_ln/fold_1/phase_C.pt \
    trav=True traj=True > log/EBV_finetune_avaj.out 2>&1 &