#!/bin/sh
# Uploads public folder to S3
aws s3 cp ./weights/production s3://wordle-ai/production --acl public-read --cache-control max-age=0