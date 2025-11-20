#!/bin/bash

# Stop script for FEMNIST distributed training

PID_FILE="/tmp/federatedscope_femnist_pids.txt"

echo "============================================"
echo "Stopping FederatedScope FEMNIST Distributed Training"
echo "============================================"

# Kill processes from PID file
if [ -f "$PID_FILE" ]; then
    echo "Found PID file, stopping tracked processes..."
    killed_count=0
    while read pid; do
        if ps -p $pid > /dev/null 2>&1; then
            echo "  Killing process $pid"
            kill -9 $pid 2>/dev/null || true
            killed_count=$((killed_count + 1))
        else
            echo "  Process $pid already stopped"
        fi
    done < "$PID_FILE"
    rm -f "$PID_FILE"
    echo "Stopped $killed_count tracked processes"
else
    echo "No PID file found"
fi

# Also kill any remaining federatedscope processes related to FEMNIST (cleanup)
echo ""
echo "Checking for remaining FederatedScope FEMNIST processes..."
remaining=$(ps aux | grep "federatedscope/main.py.*femnist" | grep -v grep | wc -l)

if [ $remaining -gt 0 ]; then
    echo "Found $remaining remaining processes, cleaning up..."
    pkill -9 -f "federatedscope/main.py.*femnist" 2>/dev/null || true
    sleep 1
    echo "Cleanup complete"
else
    echo "No remaining processes found"
fi

echo ""
echo "============================================"
echo "All FederatedScope FEMNIST processes stopped"
echo "============================================"