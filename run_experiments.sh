#!/bin/bash

WEBHOOK_URL="https://kuleuven.webhook.office.com/webhookb2/626b255b-8871-4cec-9dd6-486811dd7dfb@3973589b-9e40-4eb5-800e-b0b6383d1621/IncomingWebhook/69061b2a3cd94b95aa5fc9e1d6ef5a12/bacc6a49-b24e-44a3-b54b-abc73e49688c"

augment=('False' 'True')
train_subset_size=(1000 2000 5000 10000 20000)
train_subset_seed=(482 743 267 899)
scenario=('scratch' 'pretrained_imagenet' 'pretrained_inaturalist') #pretrained_inaturalist
model_name='tf_efficientnetv2_m.in21k_ft_in1k'

# tf_efficientnetv2_l.in21k_ft_in1k
# tf_efficientnetv2_m.in21k_ft_in1k
# tf_efficientnetv2_s.in21k_ft_in1k

len_augment=$(( ${#augment[*]} - 1 ))
len_train_subset_size=$(( ${#train_subset_size[*]} - 1 ))
len_scenarios=$(( ${#scenario[*]} - 1 ))

# Start with experiments involving subset datasets
for i in $(seq 0 $len_augment); do
    for j in $(seq 0 $len_train_subset_size); do
        for k in $(seq 0 $len_scenarios); do
        output_file=$(mktemp)
        python train.py --enable_wandb True \
            --augment ${augment[$i]} \
            --train_subset_size ${train_subset_size[$j]} \
            --scenario ${scenario[$k]} \
            --model_name $model_name 2> >(tee -a >(grep -v '^\s*$' | grep -vi 'warning' > "$output_file") >&2) \
            --train_subset_seed 743
        # Check the exit code of the Python script
        if [ $? -ne 0 ]; then
            OUTPUT=$(<"$output_file")
            ERROR_MESSAGE="COMBINATION: augment=${augment[$i]}, train_subset_size=${train_subset_size[$j]}, scenario=${scenario[$k]}"
            ERROR_MESSAGE+=" ------> ERROR: $OUTPUT"
            PAYLOAD=$(jq -n --arg message "$ERROR_MESSAGE" '{"text": $message}')
            curl -X POST -H "Content-Type: application/json" -d "$PAYLOAD" "$WEBHOOK_URL"
        fi
        rm -f "$output_file"
        done
    done
done

# Continue with the entire dataset
for i_ in $(seq 0 $len_augment); do
    for j_ in $(seq 0 $len_scenarios); do
    output_file=$(mktemp)
    python train.py --enable_wandb True \
        --augment ${augment[$i_]} \
        --scenario ${scenario[$j_]} \
        --model_name $model_name 2> >(tee -a >(grep -v '^\s*$' | grep -vi 'warning' > "$output_file") >&2)
    # Check the exit code of the Python script
    if [ $? -ne 0 ]; then
        OUTPUT=$(<"$output_file")
        ERROR_MESSAGE="COMBINATION: augment=${augment[$i]}, train_subset_size=None, scenario=${scenario[$k]}"
        ERROR_MESSAGE+=" ------> ERROR: $OUTPUT"
        PAYLOAD=$(jq -n --arg message "$ERROR_MESSAGE" '{"text": $message}')
        curl -X POST -H "Content-Type: application/json" -d "$PAYLOAD" "$WEBHOOK_URL"
    fi
    rm -f "$output_file"
    done
done



END_MESSAGE="Your run_experiments.sh script has finished running."
curl -X POST -H "Content-Type: application/json" -d "{\"text\":\"$END_MESSAGE\"}" $WEBHOOK_URL
