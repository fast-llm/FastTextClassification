#!/usr/bin/env python
# coding=utf-8
"""
@author: RongkangXiong
@contact: gunicorn_conf.py
@file: 
@date: 2024/05/15 15:28:48
@desc: 
"""
import os
import uvicorn

workers = os.environ.get('WORKERS', 1)
worker_class = 'uvicorn.workers.UvicornWorker'
bind = f"0.0.0.0:{os.getenv('PORT', 9090)}"