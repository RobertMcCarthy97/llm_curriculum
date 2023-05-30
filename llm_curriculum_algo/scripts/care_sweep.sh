#!/bin/bash

# One-hot
python ../sb3_train.py --override_hparams --exp_name one_hot-care-n1 --features_extractor care --care_n_experts 1
python ../sb3_train.py --override_hparams --exp_name one_hot-care-n2 --features_extractor care --care_n_experts 2
python ../sb3_train.py --override_hparams --exp_name one_hot-care-n3 --features_extractor care --care_n_experts 3

# language
python ../sb3_train.py --override_hparams --exp_name language-care-n2 --features_extractor care --care_n_experts 2 --use_language_goals
python ../sb3_train.py --override_hparams --exp_name language-care-n3 --features_extractor care --care_n_experts 3 --use_language_goals
