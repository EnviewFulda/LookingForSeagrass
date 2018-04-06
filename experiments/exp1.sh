##################
#---Experiment1---
##################
# adapt the following three paths
GRAPH="/path/to/classify_image_graph_def.pb"
FOLDER_ROOT="/path/to/datasetroot/dataset"
OUTPUT_PATH="/path/to/output/results"


FOLDER_IMAGES="/images"
FOLDER_GROUND_TRUTH="/ground-truth"
EVAL_TRAIN="/train.json"
EVAL_TEST="/test.json"
EVAL_VALIDATE="/validate.json"
DEPTH_MIN=0.0
DEPTH_MAX=10.0
PATCH_SIZE=240

MAIN_PROGRAM="../main.py"

python3 $MAIN_PROGRAM \
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
--output "$OUTPUT_PATH/exp1-cnn-sp-slic.json" \
--depth_min $DEPTH_MIN \
--depth_max $DEPTH_MAX \

python3 $MAIN_PROGRAM \
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
--output "$OUTPUT_PATH/exp1-hog-sp-slic.json" \
--depth_min $DEPTH_MIN \
--depth_max $DEPTH_MAX \

python3 $MAIN_PROGRAM \
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
--output "$OUTPUT_PATH/exp1-lbp-sp-slic.json" \
--depth_min $DEPTH_MIN \
--depth_max $DEPTH_MAX \


python3 $MAIN_PROGRAM \
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
--output "$OUTPUT_PATH/exp1-cnn-rp.json" \
--depth_min $DEPTH_MIN \
--depth_max $DEPTH_MAX \

python3 $MAIN_PROGRAM \
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
--output "$OUTPUT_PATH/exp1-hog-rp.json" \
--depth_min $DEPTH_MIN \
--depth_max $DEPTH_MAX \

python3 $MAIN_PROGRAM \
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
--output "$OUTPUT_PATH/exp1-lbp-rp.json" \
--depth_min $DEPTH_MIN \
--depth_max $DEPTH_MAX \


##################

python3 $MAIN_PROGRAM \
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
--output "$OUTPUT_PATH/exp1-hog-sp-cw.json" \
--depth_min $DEPTH_MIN \
--depth_max $DEPTH_MAX \

##################

python3 $MAIN_PROGRAM \
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
--output "$OUTPUT_PATH/exp1-lbp-sp-cw.json" \
--depth_min $DEPTH_MIN \
--depth_max $DEPTH_MAX \


python3 $MAIN_PROGRAM \
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
--output "$OUTPUT_PATH/exp1-cnn-sp-cw.json" \
--depth_min $DEPTH_MIN \
--depth_max $DEPTH_MAX \


