#!/bin/bash

# AI Image Search - Startup Script
# This script starts both the backend and frontend servers

echo "🚀 Starting AI Image Search Application..."
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Ollama is running
echo "Checking Ollama..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${RED}❌ Ollama is not running!${NC}"
    echo "Please start Ollama first:"
    echo "  ollama serve"
    exit 1
fi
echo -e "${GREEN}✅ Ollama is running${NC}"
echo ""

# Get project root
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup EXIT INT TERM

# Start Backend
echo -e "${BLUE}Starting Backend API (Port 8000)...${NC}"
cd "$PROJECT_ROOT/backend"
python3 api.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 10

# Check if backend started successfully
if ! curl -s http://localhost:8000/ > /dev/null 2>&1; then
    echo -e "${RED}❌ Backend failed to start${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Backend running at http://localhost:8000${NC}"
echo ""

# Start Frontend
echo -e "${BLUE}Starting Frontend (Port 3000)...${NC}"
cd "$PROJECT_ROOT/frontend"
npm run dev &
FRONTEND_PID=$!

# Wait for frontend to start
sleep 5

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}🎉 Application is ready!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Frontend: ${BLUE}http://localhost:3000${NC}"
echo -e "Backend:  ${BLUE}http://localhost:8000${NC}"
echo -e "API Docs: ${BLUE}http://localhost:8000/docs${NC}"
echo ""
echo "Press Ctrl+C to stop all servers"
echo ""

# Wait indefinitely
wait