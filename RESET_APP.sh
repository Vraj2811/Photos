#!/bin/bash

# RESET_APP.sh
# This script clears all data from the AI Image Search system.
# WARNING: This will delete all uploaded images, database records, and AI indexes.

echo "⚠️  Resetting AI Image Search Application..."

# 1. Remove the database
if [ -f "image_search.db" ]; then
    echo "🗑️  Deleting database..."
    rm image_search.db
fi

# 2. Clear the uploads folder
if [ -d "uploads" ]; then
    echo "🗑️  Clearing uploads folder..."
    rm -rf uploads
fi
mkdir -p uploads

# 3. Clear the thumbnails folder
if [ -d "backend/thumbnails" ]; then
    echo "🗑️  Clearing thumbnails folder..."
    rm -rf backend/thumbnails
fi
mkdir -p backend/thumbnails

# 4. Clear the FAISS indexes folder
if [ -d "faiss_indexes" ]; then
    echo "🗑️  Clearing FAISS indexes..."
    rm -rf faiss_indexes
fi
mkdir -p faiss_indexes

echo "✅ Reset complete! The application is now in a fresh state."
echo "You can now start the app using your startup script."
