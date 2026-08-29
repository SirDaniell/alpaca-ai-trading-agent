#!/bin/bash
# Quick start script for the Alpaca Trading Agent backend

set -e

BACKEND_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BACKEND_DIR"

echo "=========================================="
echo "Alpaca Trading Agent - Backend Setup"
echo "=========================================="
echo ""

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate venv
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Install/upgrade dependencies
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Edit .env and add your Alpaca API credentials:"
    echo "   ALPACA_API_KEY=your_key_here"
    echo "   ALPACA_SECRET_KEY=your_secret_here"
    echo ""
fi

# Check database
echo "🗄️  Checking database configuration..."
if ! command -v psql &> /dev/null; then
    echo "⚠️  PostgreSQL client not found. You'll need to set up PostgreSQL manually or use Docker:"
    echo "   docker-compose up postgres"
    echo ""
else
    echo "✓ PostgreSQL available"
fi

# Initialize database
echo "🔨 Initializing database..."
python -c "from app.db.connection import init_db; init_db()" 2>/dev/null || echo "⚠️  Database initialization skipped (DB not running yet)"

echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env with your Alpaca API credentials"
echo "2. Start PostgreSQL: docker-compose up postgres"
echo "3. Run the server: python -m uvicorn app.main:app --reload"
echo "4. Test the API: curl http://localhost:8000/health"
echo ""
echo "For more info, see: README.md"
