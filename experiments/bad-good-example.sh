##################
#---Experiment2---
##################
GRAPH="/home/jaeger/papers/2018-oceans-seagrass/classify_image_graph_def.pb"
FOLDER_ROOT="/home/jaeger/papers/2018-oceans-seagrass/dataset"

FOLDER_IMAGES="/images"
FOLDER_GROUND_TRUTH="/ground-truth"
EVAL_TRAIN="/train.json"
EVAL_TEST="/bad-good-example.json"
EVAL_VALIDATE="/validate.json"
DEPTH_MIN=0.0
DEPTH_MAX=2.0

MAIN_PROGRAM="../main.py"

PATCH_SIZE=120
python3 $MAIN_PROGRAM \
--pattern SP_SLIC \
--features cnn \
--graph $GRAPH \
--mode "debug" \
--folder_root $FOLDER_ROOT \
--folder_images $FOLDER_IMAGES \
--folder_ground_truth $FOLDER_GROUND_TRUTH \
--eval_test $EVAL_TEST \
--eval_train $EVAL_TRAIN \
--eval_validate $EVAL_VALIDATE \
--patch_size_width $PATCH_SIZE \
--patch_size_height $PATCH_SIZE \
--output "/home/jaeger/papers/2018-oceans-seagrass/results/bad-good-cnn-sp-slic-120-0-2.json" \
--depth_min $DEPTH_MIN \
--depth_max $DEPTH_MAX \

python3 $MAIN_PROGRAM \
--pattern RP \
--features cnn \
--graph $GRAPH \
--mode "debug" \
--folder_root $FOLDER_ROOT \
--folder_images $FOLDER_IMAGES \
--folder_ground_truth $FOLDER_GROUND_TRUTH \
--eval_test $EVAL_TEST \
--eval_train $EVAL_TRAIN \
--eval_validate $EVAL_VALIDATE \
--patch_size_width $PATCH_SIZE \
--patch_size_height $PATCH_SIZE \
--output "/home/jaeger/papers/2018-oceans-seagrass/results/bad-good-cnn-rp-120-0-2.json" \
--depth_min $DEPTH_MIN \
--depth_max $DEPTH_MAX \

python3 $MAIN_PROGRAM \
--pattern SP_CW \
--features cnn \
--graph $GRAPH \
--mode "debug" \
--folder_root $FOLDER_ROOT \
--folder_images $FOLDER_IMAGES \
--folder_ground_truth $FOLDER_GROUND_TRUTH \
--eval_test $EVAL_TEST \
--eval_train $EVAL_TRAIN \
--eval_validate $EVAL_VALIDATE \
--patch_size_width $PATCH_SIZE \
--patch_size_height $PATCH_SIZE \
--output "/home/jaeger/papers/2018-oceans-seagrass/results/bad-good-cnn-sp-cw-120-0-2.json" \
--depth_min $DEPTH_MIN \
--depth_max $DEPTH_MAX \

