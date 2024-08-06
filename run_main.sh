#!/bin/bash

WEBHOOK_URL="https://kuleuven.webhook.office.com/webhookb2/626b255b-8871-4cec-9dd6-486811dd7dfb@3973589b-9e40-4eb5-800e-b0b6383d1621/IncomingWebhook/69061b2a3cd94b95aa5fc9e1d6ef5a12/bacc6a49-b24e-44a3-b54b-abc73e49688c"

output_file=$(mktemp)
python main.py 2> >(tee -a >(grep -v '^\s*$' | grep -vi 'warning' > "$output_file") >&2)
# Check the exit code of the Python script
ERROR_MESSAGE=""
if [ $? -ne 0 ]; then
    OUTPUT=$(<"$output_file")
    ERROR_MESSAGE+="ERROR: $OUTPUT"
    PAYLOAD=$(jq -n --arg message "$ERROR_MESSAGE" '{"text": $message}')
    curl -X POST -H "Content-Type: application/json" -d "$PAYLOAD" "$WEBHOOK_URL"
fi
rm -f "$output_file"

END_MESSAGE="Your main script has finished running."
curl -X POST -H "Content-Type: application/json" -d "{\"text\":\"$END_MESSAGE\"}" $WEBHOOK_URL