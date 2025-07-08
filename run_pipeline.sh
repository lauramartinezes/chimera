#!/bin/bash
WEBHOOK_URL="https://kuleuven.webhook.office.com/webhookb2/626b255b-8871-4cec-9dd6-486811dd7dfb@3973589b-9e40-4eb5-800e-b0b6383d1621/IncomingWebhook/19193a3061c54c44804b25e534c58362/bacc6a49-b24e-44a3-b54b-abc73e49688c/V2DCJVyA8y7XioAjg-m1aY_lRTnm-krjfryBo5hLOX1Ng1"

declare -A files_with_args=(
  ["a00_split_dataset_raw_dfs.py"]=""
  ["a01_prepare_dataset_raw_dfs.py"]=""
  ["a02_train_test_classifier.py"]=""
  ["a03_prepare_data_swap_dfs.py"]="" 
  ["a04_generate_umaps.py"]=""
  ["a06_evaluate_outlier_detection.py"]=""
  ["a07_clean_dfs.py"]="" 
  ["a08_analyze_meas_label_noise.py"]="" 
  ["a02_train_test_classifier.py --experiments all_cases"]=""
)

for entry in "${!files_with_args[@]}"; do
    file="${entry%% *}"
    args="${entry#"$file"}"

    echo "Running $file$args"
    output_file=$(mktemp)
    python "$file" $args 2> >(tee -a >(grep -v '^\s*$' | grep -vi 'warning' > "$output_file") >&2)
    
    ERROR_MESSAGE=""
    if [ $? -ne 0 ]; then
        OUTPUT=$(<"$output_file")
        ERROR_MESSAGE+="ERROR: $OUTPUT"
        PAYLOAD=$(jq -n --arg message "$ERROR_MESSAGE" '{"text": $message}')
        curl -X POST -H "Content-Type: application/json" -d "$PAYLOAD" "$WEBHOOK_URL"
        echo "$file failed. Exiting."
        exit 1
    fi
    rm -f "$output_file"
done


END_MESSAGE="All scripts executed successfully!"
curl -X POST -H "Content-Type: application/json" -d "{\"text\":\"$END_MESSAGE\"}" $WEBHOOK_URL

echo $END_MESSAGE