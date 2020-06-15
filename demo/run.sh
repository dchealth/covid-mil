folder=`dirname $0`
echo "Training folder: $folder"

train_num=4000
nepochs=100
test_every=10
batch=256
target_key=ICU
plan=`readlink -f $folder/plan.pkl`

source $UTILS_FILE
echo "Wait for idle GPU..."
gpu=0
echo "Find idle gpu: $gpu"
export CUDA_VISIBLE_DEVICES=$gpu

python covid_train.py \
--train_lib 'yes' --val_lib 'yes' \
--plan $plan \
--weights 0.8 \
--k 4 \
--target_key $target_key \
--output $folder --batch_size $batch --nepochs ${nepochs}  \
--test_every ${test_every} 2>&1 | tee $folder/train.py.log

python covid_test.py \
--plan $plan  \
--target_key $target_key \
--output $folder --batch_size $batch \
--model ${folder}/checkpoint_best.pth \
--dataset valid 2>&1 | tee $folder/test_valid.py.log

python covid_test.py \
--plan $plan  \
--target_key $target_key \
--output $folder --batch_size $batch \
--model ${folder}/checkpoint_best.pth \
--dataset train 2>&1 | tee $folder/test_train.py.log

plan=`readlink -f $folder/test.pkl`
python covid_test.py \
--plan $plan  \
--target_key $target_key \
--output $folder --batch_size $batch \
--model ${folder}/checkpoint_best.pth \
--dataset test 2>&1 | tee $folder/test_test.py.log