#!/bin/bash
# ==============================================================================
# DeliverAI - AWS S3 DVC Remote Configuration Script (CO1)
# ==============================================================================
# This script sets up an AWS S3 bucket as the remote storage backend for DVC,
# allowing models and training datasets to be versioned and synchronized to the cloud.
# ==============================================================================

set -e

BUCKET_NAME="${1:-deliverai-mlops-bucket}"
AWS_REGION="${2:-us-east-1}"

echo "=== DeliverAI S3 DVC Remote Setup ==="
echo "Target S3 Bucket: s3://$BUCKET_NAME/dvc-storage"
echo "AWS Region:       $AWS_REGION"

# 1. Check AWS CLI authentication
if ! command -v aws &> /dev/null; then
    echo "Error: AWS CLI is not installed. Please install and run 'aws configure'."
    exit 1
fi

echo "Verifying AWS credentials..."
aws sts get-caller-identity

# 2. Create S3 Bucket if it does not exist
echo "Ensuring S3 bucket exists..."
if [ "$AWS_REGION" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$BUCKET_NAME" --region "$AWS_REGION" || true
else
    aws s3api create-bucket --bucket "$BUCKET_NAME" --region "$AWS_REGION" \
        --create-bucket-configuration LocationConstraint="$AWS_REGION" || true
fi

# Enable bucket versioning for data lineage
aws s3api put-bucket-versioning --bucket "$BUCKET_NAME" \
    --versioning-configuration Status=Enabled

# 3. Add S3 Remote in DVC
echo "Configuring DVC remote..."
dvc remote add -d -f s3remote "s3://$BUCKET_NAME/dvc-storage"
dvc remote modify s3remote region "$AWS_REGION"

echo "=== DVC Remote Configuration Completed ==="
dvc remote list

echo ""
echo "To push versioned models and dataset to S3:"
echo "  dvc push -r s3remote"
echo ""
echo "To pull artifacts onto EC2 or another machine:"
echo "  dvc pull -r s3remote"

# DeliverAI
