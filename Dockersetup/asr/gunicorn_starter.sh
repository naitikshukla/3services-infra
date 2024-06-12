#!/bin/sh

gunicorn --chdir app app:asr_app -w 2 --threads 2 -b 0.0.0.0:8003