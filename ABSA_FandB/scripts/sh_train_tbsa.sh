python3 ./run_tbsa.py \
      --approach phobert_mixlayer \
      --mix_count 4 \
      --mix_type HSUM \
      --data_dir ../data/FMCG/ver0.3/split/TBSA/ \
      --device cuda \
      --max_length 256 \
      --epochs 10 \
      --pool_type sum \
      --lr 3e-5 \
      --gradient_accumulation_steps 1 \
      --do_lower_case \
      --replace_term_with_mask \
      --train_batch_size 8 \
      --test_batch_size 8