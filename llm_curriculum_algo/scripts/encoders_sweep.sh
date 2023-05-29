#!/bin/bash

# Compare one-hot vs language
python ../sb3_train.py --override_hparams --exp_name one_hot-standard_encoder --features_extractor custom --share_features_extractor
python ../sb3_train.py --override_hparams --exp_name language-standard_encoder --use_language_goals --features_extractor custom --share_features_extractor

# Compare Extractor share vs no share
python ../sb3_train.py --override_hparams --exp_name language-standard_encoder-no_share --use_language_goals --features_extractor custom

# Compare FiLM one-hot vs language
python ../sb3_train.py --override_hparams --exp_name one_hot-film_encoder --features_extractor film --share_features_extractor
python ../sb3_train.py --override_hparams --exp_name language-film_encoder --use_language_goals --features_extractor film --share_features_extractor