##################
#---Experiment1---
##################
DEPTH_MIN=0.0
DEPTH_MAX=10.0
EVAL_TRAIN="/quicktest/train.json"
EVAL_TEST="/quicktest/test.json"
EVAL_VALIDATE="/quicktest/validate.json"


python3 ../src/seagrass/main.py \
--pattern SP_SLIC \
--features cnn \
--graph "/home/thomas/cnn_feature_extractor/tf_inception/classify_image_graph_def.pb" \
--mode dontcare \
--folder_root "/home/thomas/experiments/dataset" \
--folder_images "/images" \
--folder_ground_truth "/ground-truth" \
--eval_test $EVAL_TEST \
--eval_train $EVAL_TRAIN \
--eval_validate $EVAL_VALIDATE \
--patch_size_width 120 \
--patch_size_height 120 \
--output "/home/thomas/experiments/results/exp1-cnn-sp-slic.json" \
--depth_min $DEPTH_MIN \
--depth_max $DEPTH_MAX \


python3 ../src/seagrass/main.py \
--pattern SP_CW \
--features cnn \
--graph "/home/thomas/cnn_feature_extractor/tf_inception/classify_image_graph_def.pb" \
--mode dontcare \
--folder_root "/home/thomas/experiments/dataset" \
--folder_images "/images" \
--folder_ground_truth "/ground-truth" \
--eval_test $EVAL_TEST \
--eval_train $EVAL_TRAIN \
--eval_validate $EVAL_VALIDATE \
--patch_size_width 120 \
--patch_size_height 120 \
--output "/home/thomas/experiments/results/exp1-cnn-sp-cw.json" \
--depth_min $DEPTH_MIN \
--depth_max $DEPTH_MAX \


python3 ../src/seagrass/main.py \
--pattern RP \
--features cnn \
--graph "/home/thomas/cnn_feature_extractor/tf_inception/classify_image_graph_def.pb" \
--mode dontcare \
--folder_root "/home/thomas/experiments/dataset" \
--folder_images "/images" \
--folder_ground_truth "/ground-truth" \
--eval_test $EVAL_TEST \
--eval_train $EVAL_TRAIN \
--eval_validate $EVAL_VALIDATE \
--patch_size_width 120 \
--patch_size_height 120 \
--output "/home/thomas/experiments/results/exp1-cnn-rp.json" \
--depth_min $DEPTH_MIN \
--depth_max $DEPTH_MAX \


##################


python3 ../src/seagrass/main.py \
--pattern SP_SLIC \
--features hog \
--graph "/home/thomas/cnn_feature_extractor/tf_inception/classify_image_graph_def.pb" \
--mode dontcare \
--folder_root "/home/thomas/experiments/dataset" \
--folder_images "/images" \
--folder_ground_truth "/ground-truth" \
--eval_test $EVAL_TEST \
--eval_train $EVAL_TRAIN \
--eval_validate $EVAL_VALIDATE \
--patch_size_width 120 \
--patch_size_height 120 \
--output "/home/thomas/experiments/results/exp1-hog-sp-slic.json" \
--depth_min $DEPTH_MIN \
--depth_max $DEPTH_MAX \


python3 ../src/seagrass/main.py \
--pattern SP_CW \
--features hog \
--graph "/home/thomas/cnn_feature_extractor/tf_inception/classify_image_graph_def.pb" \
--mode dontcare \
--folder_root "/home/thomas/experiments/dataset" \
--folder_images "/images" \
--folder_ground_truth "/ground-truth" \
--eval_test $EVAL_TEST \
--eval_train $EVAL_TRAIN \
--eval_validate $EVAL_VALIDATE \
--patch_size_width 120 \
--patch_size_height 120 \
--output "/home/thomas/experiments/results/exp1-hog-sp-cw.json" \
--depth_min $DEPTH_MIN \
--depth_max $DEPTH_MAX \


python3 ../src/seagrass/main.py \
--pattern RP \
--features hog \
--graph "/home/thomas/cnn_feature_extractor/tf_inception/classify_image_graph_def.pb" \
--mode dontcare \
--folder_root "/home/thomas/experiments/dataset" \
--folder_images "/images" \
--folder_ground_truth "/ground-truth" \
--eval_test $EVAL_TEST \
--eval_train $EVAL_TRAIN \
--eval_validate $EVAL_VALIDATE \
--patch_size_width 120 \
--patch_size_height 120 \
--output "/home/thomas/experiments/results/exp1-hog-rp.json" \
--depth_min $DEPTH_MIN \
--depth_max $DEPTH_MAX \


##################


python3 ../src/seagrass/main.py \
--pattern SP_SLIC \
--features lbp \
--graph "/home/thomas/cnn_feature_extractor/tf_inception/classify_image_graph_def.pb" \
--mode dontcare \
--folder_root "/home/thomas/experiments/dataset" \
--folder_images "/images" \
--folder_ground_truth "/ground-truth" \
--eval_test $EVAL_TEST \
--eval_train $EVAL_TRAIN \
--eval_validate $EVAL_VALIDATE \
--patch_size_width 120 \
--patch_size_height 120 \
--output "/home/thomas/experiments/results/exp1-lbp-sp-slic.json" \
--depth_min $DEPTH_MIN \
--depth_max $DEPTH_MAX \


python3 ../src/seagrass/main.py \
--pattern SP_CW \
--features lbp \
--graph "/home/thomas/cnn_feature_extractor/tf_inception/classify_image_graph_def.pb" \
--mode dontcare \
--folder_root "/home/thomas/experiments/dataset" \
--folder_images "/images" \
--folder_ground_truth "/ground-truth" \
--eval_test $EVAL_TEST \
--eval_train $EVAL_TRAIN \
--eval_validate $EVAL_VALIDATE \
--patch_size_width 120 \
--patch_size_height 120 \
--output "/home/thomas/experiments/results/exp1-lbp-sp-cw.json" \
--depth_min $DEPTH_MIN \
--depth_max $DEPTH_MAX \


python3 ../src/seagrass/main.py \
--pattern RP \
--features lbp \
--graph "/home/thomas/cnn_feature_extractor/tf_inception/classify_image_graph_def.pb" \
--mode dontcare \
--folder_root "/home/thomas/experiments/dataset" \
--folder_images "/images" \
--folder_ground_truth "/ground-truth" \
--eval_test $EVAL_TEST \
--eval_train $EVAL_TRAIN \
--eval_validate $EVAL_VALIDATE \
--patch_size_width 120 \
--patch_size_height 120 \
--output "/home/thomas/experiments/results/exp1-lbp-rp.json" \
--depth_min $DEPTH_MIN \
--depth_max $DEPTH_MAX \
