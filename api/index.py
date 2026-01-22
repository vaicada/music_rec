"""
Vercel Serverless Function Entry Point for Music Recommender API.

This file serves as the entry point for Vercel's Python runtime.
It wraps the FastAPI application from web_app/app.py to make it compatible
with Vercel's serverless function architecture.
"""

import sys
import os

# Add parent directory to path to import web_app
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import the FastAPI app
from web_app.app import app

# For Vercel, we need to export the app
# Vercel will look for a variable called "app" or a function called "handler"
handler = app
