#!/bin/bash
MODEL_ROOT=/home/jhu/dev/models/embs \
    uvicorn manager.app:app --host 0.0.0.0 --port 18000