# ABSA_F&B
**Training a model**
```
python3 [task_run_file] --approach [model_name] ...
All parameters and models' name are mentioned in file run_tbsa.py and run_acd_acsa.py
```
Example:
```
python3 ./run_acd_acsa.py \
          --approach phobert \
          --data_dir ../data/FandB/ver0.1/split/ACSA \
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
    - Infer time: `4.049ms` per example
    - Accuracy: `81.399` || F1-score: `75.810`
- ACD + ACSA
    - Infer time: `7.960ms` per example
    - Accuracy: `95.405` || macro_F1: `79.086` (consider None as a class)
    - Accuracy: `67.947` || macro_F1: `54.630` (correct None prediction is eleminated)
    - strict_acc_acd: `87.568`
    - strict_acc_acd_acsa: `83.063`
- For more details: [results](https://lab.admicro.vn/corenlp/aspect-based-sentiment-analysis/-/blob/doc/ABSA_results.xlsx)

**Parameters in use**
- TBSA:
    - approach: `phobert_mixlayer`
    - max_length: `256`
    - stride: `4`
    - epochs: `10`
    - lr: `3e-5`
    - pool_type: `max`
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
    - pool_type: `sum`
    - do_lower_case: `True`
