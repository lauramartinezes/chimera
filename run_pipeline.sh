#!/bin/bash
WEBHOOK_URL="https://kuleuven.webhook.office.com/webhookb2/626b255b-8871-4cec-9dd6-486811dd7dfb@3973589b-9e40-4eb5-800e-b0b6383d1621/IncomingWebhook/19193a3061c54c44804b25e534c58362/bacc6a49-b24e-44a3-b54b-abc73e49688c/V2DCJVyA8y7XioAjg-m1aY_lRTnm-krjfryBo5hLOX1Ng1"

files=(
  "a00_split_dataset_raw_dfs.py"
  "a01_prepare_dataset_raw_dfs.py"
  "a02_train_test_classifier.py"
  "a03_prepare_data_swap_dfs.py" 
  "a04_generate_umaps.py"
  "a06_evaluate_outlier_detection.py"
  "a07_clean_dfs.py" 
  "a08_analyze_meas_label_noise.py" 
)

for file in "${files[@]}"; do
    echo "Running $file"
    output_file=$(mktemp)
    python "$file" 2> >(tee -a >(grep -v '^\s*$' | grep -vi 'warning' > "$output_file") >&2)
    # Check the exit code of the Python script
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

echo "Running a02_train_test_classifier.py for all cases"
output_file=$(mktemp)
    python "a02_train_test_classifier.py" --experiments all_cases 2> >(tee -a >(grep -v '^\s*$' | grep -vi 'warning' > "$output_file") >&2)
    # Check the exit code of the Python script
    ERROR_MESSAGE=""
    if [ $? -ne 0 ]; then
        OUTPUT=$(<"$output_file")
        ERROR_MESSAGE+="ERROR: $OUTPUT"
        PAYLOAD=$(jq -n --arg message "$ERROR_MESSAGE" '{"text": $message}')
        curl -X POST -H "Content-Type: application/json" -d "$PAYLOAD" "$WEBHOOK_URL"
        echo "a02_train_test_classifier.py failed. Exiting."
        exit 1
    fi


END_MESSAGE="All scripts executed successfully!"
curl -X POST -H "Content-Type: application/json" -d "{\"text\":\"$END_MESSAGE\"}" $WEBHOOK_URL

echo $END_MESSAGE
