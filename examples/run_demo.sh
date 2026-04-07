#!/bin/bash

echo "Starting SignCheck-env FastAPI server..."
uvicorn server.main:app --port 7860 &
SERVER_PID=$!

# Give the server a few seconds to start up
sleep 3

echo "Running inference script..."
python inference.py

echo "Cleaning up..."
kill $SERVER_PID
