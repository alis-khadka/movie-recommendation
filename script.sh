#!/bin/bash

if [ "$1" == "--dev" ]; then
  uvicorn main:app --reload
elif [ "$1" == "--start" ]; then
  uvicorn main:app
else
  echo "Usage: ./script.sh [--dev | --start]"
  exit 1
fi