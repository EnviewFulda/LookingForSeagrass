##################
#---Experiment1---
##################
DEPTH_MIN=0.0
DEPTH_MAX=10.0
EVAL_TRAIN="/quicktest/train.json"
EVAL_TEST="/quicktest/test.json"
EVAL_VALIDATE="/quicktest/validate.json"

GRAPH="/home/thomas/cnn_feature_extractor/tf_inception/classify_image_graph_def.pb"
FOLDER_ROOT="/home/thomas/experiments/dataset"
FOLDER_IMAGES="/images"
FOLDER_GROUND_TRUTH="/ground-truth"
PATCH_SIZE=120


python3 ../main.py \
--pattern SP_SLIC \
--features cnn \
--graph $GRAPH \
--mode dontcare \
--folder_root $FOLDER_ROOT \
--folder_images $FOLDER_IMAGES \
--folder_ground_truth $FOLDER_GROUND_TRUTH \
--eval_test $EVAL_TEST \
--eval_train $EVAL_TRAIN \
--eval_validate $EVAL_VALIDATE \
--patch_size_width $PATCH_SIZE \
--patch_size_height $PATCH_SIZE \
--output "/home/thomas/experiments/results/exp1-cnn-sp-slic.json" \
--depth_min $DEPTH_MIN \
--depth_max $DEPTH_MAX \


python3 ../main.py \
--pattern SP_CW \
--features cnn \
--graph $GRAPH \
--mode dontcare \
--folder_root $FOLDER_ROOT \
--folder_images $FOLDER_IMAGES \
--folder_ground_truth $FOLDER_GROUND_TRUTH \
--eval_test $EVAL_TEST \
--eval_train $EVAL_TRAIN \
--eval_validate $EVAL_VALIDATE \
--patch_size_width $PATCH_SIZE \
--patch_size_height $PATCH_SIZE \
--output "/home/thomas/experiments/results/exp1-cnn-sp-cw.json" \
--depth_min $DEPTH_MIN \
--depth_max $DEPTH_MAX \


python3 ../main.py \
--pattern RP \
--features cnn \
--graph $GRAPH \
--mode dontcare \
--folder_root $FOLDER_ROOT \
--folder_images $FOLDER_IMAGES \
--folder_ground_truth $FOLDER_GROUND_TRUTH \
--eval_test $EVAL_TEST \
--eval_train $EVAL_TRAIN \
--eval_validate $EVAL_VALIDATE \
--patch_size_width $PATCH_SIZE \
--patch_size_height $PATCH_SIZE \
--output "/home/thomas/experiments/results/exp1-cnn-rp.json" \
--depth_min $DEPTH_MIN \
--depth_max $DEPTH_MAX \


##################


python3 ../main.py \
--pattern SP_SLIC \
--features hog \
--graph $GRAPH \
--mode dontcare \
--folder_root $FOLDER_ROOT \
--folder_images $FOLDER_IMAGES \
--folder_ground_truth $FOLDER_GROUND_TRUTH \
--eval_test $EVAL_TEST \
--eval_train $EVAL_TRAIN \
--eval_validate $EVAL_VALIDATE \
--patch_size_width $PATCH_SIZE \
--patch_size_height $PATCH_SIZE \
--output "/home/thomas/experiments/results/exp1-hog-sp-slic.json" \
--depth_min $DEPTH_MIN \
--depth_max $DEPTH_MAX \


python3 ../main.py \
--pattern SP_CW \
--features hog \
--graph $GRAPH \
--mode dontcare \
--folder_root $FOLDER_ROOT \
--folder_images $FOLDER_IMAGES \
--folder_ground_truth $FOLDER_GROUND_TRUTH \
--eval_test $EVAL_TEST \
--eval_train $EVAL_TRAIN \
--eval_validate $EVAL_VALIDATE \
--patch_size_width $PATCH_SIZE \
--patch_size_height $PATCH_SIZE \
--output "/home/thomas/experiments/results/exp1-hog-sp-cw.json" \
--depth_min $DEPTH_MIN \
--depth_max $DEPTH_MAX \


python3 ../main.py \
--pattern RP \
--features hog \
--graph $GRAPH \
--mode dontcare \
--folder_root $FOLDER_ROOT \
--folder_images $FOLDER_IMAGES \
--folder_ground_truth $FOLDER_GROUND_TRUTH \
--eval_test $EVAL_TEST \
--eval_train $EVAL_TRAIN \
--eval_validate $EVAL_VALIDATE \
--patch_size_width $PATCH_SIZE \
--patch_size_height $PATCH_SIZE \
--output "/home/thomas/experiments/results/exp1-hog-rp.json" \
--depth_min $DEPTH_MIN \
--depth_max $DEPTH_MAX \


##################


python3 ../main.py \
--pattern SP_SLIC \
--features lbp \
--graph $GRAPH \
--mode dontcare \
--folder_root $FOLDER_ROOT \
--folder_images $FOLDER_IMAGES \
--folder_ground_truth $FOLDER_GROUND_TRUTH \
--eval_test $EVAL_TEST \
--eval_train $EVAL_TRAIN \
--eval_validate $EVAL_VALIDATE \
--patch_size_width $PATCH_SIZE \
--patch_size_height $PATCH_SIZE \
--output "/home/thomas/experiments/results/exp1-lbp-sp-slic.json" \
--depth_min $DEPTH_MIN \
--depth_max $DEPTH_MAX \


python3 ../main.py \
--pattern SP_CW \
--features lbp \
--graph $GRAPH \
--mode dontcare \
--folder_root $FOLDER_ROOT \
--folder_images $FOLDER_IMAGES \
--folder_ground_truth $FOLDER_GROUND_TRUTH \
--eval_test $EVAL_TEST \
--eval_train $EVAL_TRAIN \
--eval_validate $EVAL_VALIDATE \
--patch_size_width $PATCH_SIZE \
--patch_size_height $PATCH_SIZE \
--output "/home/thomas/experiments/results/exp1-lbp-sp-cw.json" \
--depth_min $DEPTH_MIN \
--depth_max $DEPTH_MAX \


python3 ../main.py \
--pattern RP \
--features lbp \
--graph $GRAPH \
--mode dontcare \
--folder_root $FOLDER_ROOT \
--folder_images $FOLDER_IMAGES \
--folder_ground_truth $FOLDER_GROUND_TRUTH \
--eval_test $EVAL_TEST \
--eval_train $EVAL_TRAIN \
--eval_validate $EVAL_VALIDATE \
--patch_size_width $PATCH_SIZE \
--patch_size_height $PATCH_SIZE \
--output "/home/thomas/experiments/results/exp1-lbp-rp.json" \
--depth_min $DEPTH_MIN \
--depth_max $DEPTH_MAX \
