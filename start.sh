#!/bin/bash
export PORT=${PORT:-10000}
uvicorn app:app --host 0.0.0.0 --port $PORT
