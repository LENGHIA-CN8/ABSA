# ABSA_FMCG
**Training a model**
```
python3 [task_run_file] --approach [model_name] ...
All parameters and models' name are mentioned in file run_tbsa.py and run_acd_acsa.py
```
Example:
```
python3 ./run_tbsa.py \
      --approach phobert \
      --data_dir ../data/FandB/ver0.1/split/TBSA \
      --max_length 256 \
      --stride 4 \
      --epochs 10 \
      --pool_type sum \
      --lr 2e-5 \
      --gradient_accumulation_steps 2 \
      --do_lower_case \
      --replace_term_with_mask \
      --train_batch_size 16 \
      --test_batch_size 8 \
      --imb_weight \
      --device cuda:0
```

**Runing api**
- To run seperate services for tbsa and acd_acsa use: _tbsa_api.py_ and _acd_acsa_api.py_
- To run joint service use: _absa_api.py_ 

**Details about folders**
- API: classes, functions for API services
- trainer: trainer class to train models
- loader: process and load data functions
- models: model classes
- evaluate: metrics for evaluation
- resource: term vocabulary
- explore_data: data analysis, cleaning, process annotated data functions

**Current results on test set**
- TBSA:
    - Infer time: `9.684ms` per example (9.684 * number of terms (ms) per sentence)
    - Accuracy: `85.093` || macro-F1: `81.941`
- ACD + ACSA
    - Infer time: `3.091ms` per example (3.091 * number of terms * number of aspect categories (ms) per sentence)
    - Accuracy: `93.667` || macro-F1: `79.112` (consider None as a class)
    - Accuracy: `66.530` || macro-F1: `54.881` (correct None prediction is eleminated)
    - strict_acc_acd: `89.433`
    - strict_acc_acd_acsa: `84.848`
- For more details: [results](https://lab.admicro.vn/corenlp/aspect-based-sentiment-analysis/-/blob/doc/ABSA_results.xlsx)

**Parameters in use**
- TBSA:
    - approach: `phobert_mixlayer`
    - max_length: `256`
    - stride: `4`
    - epochs: `10`
    - lr: `3e-5`
    - pool_type: `sum`
    - do_lower_case: `True`
    - imb_weight: `True`
    - mix_count: `4`
    - mix_type: `HSUM`
- ACD + ACSA:
    - approach: `phobert`
    - max_length: `256`
    - stride: `4`
    - epochs: `10`
    - lr: `2e-5`
    - pool_type: `max`
    - do_lower_case: `True`
