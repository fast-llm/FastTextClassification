#!/bin/bash
mkdir -p ./logs

gunicorn -c src/server/gunicorn_conf.py src.server.app:app