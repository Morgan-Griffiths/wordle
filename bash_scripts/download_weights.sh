#!/bin/sh
aws s3 sync s3://wordle-ai ${PWD}/weights/production.checkpoint --no-sign-request --profile personal
