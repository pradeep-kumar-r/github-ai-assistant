#!/bin/bash

# RAG Application Quick Start Script

set -e

echo "🚀 Starting RAG Application Setup..."
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your OPENAI_API_KEY"
    echo "   Run: nano .env"
    exit 1
fi

# Check if OPENAI_API_KEY is set
if ! grep -q "OPENAI_API_KEY=sk-" .env 2>/dev/null; then
    echo "⚠️  OPENAI_API_KEY not set in .env"
    echo "   Please edit .env and add your OpenAI API key"
    echo "   Run: nano .env"
    exit 1
fi

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/postgres data/chroma

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

echo "🐳 Starting Docker containers..."
docker-compose up -d --build

echo ""
echo "✅ Application started successfully!"
echo ""
echo "📍 Access points:"
echo "   - Streamlit UI:  http://localhost:8501"
echo "   - FastAPI Docs:  http://localhost:8000/docs"
echo "   - API Root:      http://localhost:8000"
echo ""
echo "📊 View logs:"
echo "   docker-compose logs -f"
echo ""
echo "🛑 Stop application:"
echo "   docker-compose down"
echo ""
