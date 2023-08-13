import os


pool_types = ["sum", "max"]
batch_sizes = [16]
lrs = [1e-5, 2e-5, 3e-5]
# mix_types = ["HSUM", "PSUM"]
# mix_counts = [3,4]
cmds = []

for pool_type in pool_types:
    for batch_size in batch_sizes:
        for lr in lrs:
            command = "python3 ./run_acd_acsa.py \
          --approach phobert \
          --data_dir ../data/FandB/ver0.1/split/ACSA \
          --max_length 256 \
          --stride 4 \
          --epochs 10 \
          --pool_type {} \
          --lr {} \
          --gradient_accumulation_steps 2 \
          --do_lower_case \
          --replace_term_with_mask \
          --train_batch_size {} \
          --test_batch_size 8 \
          --device cuda:0".format(pool_type, lr, batch_size)
            cmds.append(command)

for cmd in cmds:
    print("Running", cmd)
    os.system(cmd)
