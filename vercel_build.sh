#!/bin/bash
echo "🚀 BUILD STARTED"
echo "📂 Current Directory: $(pwd)"

# Install dependencies based on file location
if [ -f "requirements.txt" ]; then
    echo "📦 Found requirements.txt at root. Installing..."
    pip install -r requirements.txt
else
    echo "⚠️ requirements.txt not found at root!"
    ls -la
fi

if [ -f "web_app/requirements-deploy.txt" ]; then
    echo "📦 Found web_app/requirements-deploy.txt. Installing..."
    pip install -r web_app/requirements-deploy.txt
else
    echo "⚠️ web_app/requirements-deploy.txt not found!"
    ls -la web_app
fi

echo "✅ BUILD FINISHED"
