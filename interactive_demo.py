"""
Interactive Demo - Terminal-based Music Recommendation Interface.

================================================================================
PURPOSE:
================================================================================
Provides a command-line interface (CLI) for testing the music recommendation
system. Users can interact with the system to:
1. Search for similar songs by song name (and optional artist)
2. Get recommendations by mood (Happy, Sad, Energetic, Calm, Angry)
3. Get recommendations by context (Party, Workout, Study, Relax, Driving)

================================================================================
FILE STRUCTURE:
================================================================================
- Helper functions: print_separator(), print_header(), format_song()
- main(): Main entry point with interactive loop
  - Initializes MusicRecommendationEngine
  - Loads model, FAISS index, and song data
  - Presents menu and handles user input
  - Calls engine methods based on user choice

================================================================================
RELATED FILES:
================================================================================
- hybrid_music_engine/inference.py: Contains MusicRecommendationEngine class
- hybrid_music_engine/config.py: Configuration settings (paths, hyperparameters)
- models/best_model.pth: Trained PyTorch model weights
- models/faiss_index.bin: FAISS similarity search index
- data/processed/train.csv: Song database with lyrics and features

================================================================================
USAGE:
================================================================================
    python interactive_demo.py

Then follow the on-screen menu to search for songs or get recommendations.

================================================================================
Author: Graduation Project
Created: 2026-01-11
================================================================================
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from hybrid_music_engine import get_config
from hybrid_music_engine.inference import MusicRecommendationEngine

def print_separator():
    """Print a visual separator line for CLI output."""
    print("-" * 60)


def print_header(text: str):
    """
    Print a formatted header with decorative borders.
    
    Args:
        text: The header text to display.
    """
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)

def format_song(song):
    """Format song info for display."""
    return (
        f"[*] {song['song']} - {song['artist']}\n"
        f"   Genre: {song['genre']} | Emotion: {song['emotion']} | Sim: {song.get('similarity', 'N/A')}"
    )

def main():
    print_header("MUSIC RECOMMENDATION SYSTEM - INTERACTIVE DEMO")
    
    # 1. Initialize
    print("Initializing engine... (this may take a minute)")
    config = get_config()
    engine = MusicRecommendationEngine(config)
    
    # Check paths
    model_path = os.path.join(config.paths.model_dir, "best_model.pth")
    index_path = config.paths.faiss_index_path
    data_path = os.path.join(config.paths.processed_data_dir, "dataset.csv") # Using full dataset for lookup
    # Or maybe train.csv/test.csv combined? 
    # Usually we want the dataset that corresponds to the index. 
    # In build_faiss_index.py, we created 'song_metadata.csv'. Let's use that if available.
    
    # Use train.csv which has full data (lyrics, audio features) for encoding
    # song_metadata.csv only has basic info (song_name, artist, genre, emotion)
    data_path = os.path.join(config.paths.processed_data_dir, "train.csv")
    if not os.path.exists(data_path):
        # Fallback to metadata if train.csv not found
        data_path = os.path.join(config.paths.model_dir, "song_metadata.csv")

    try:
        engine.load_model(model_path)
        engine.load_index(index_path)
        engine.load_song_data(data_path)
    except Exception as e:
        print(f"\n❌ Error initializing engine: {e}")
        print("Please ensure you have trained the model and built the FAISS index.")
        return

    print("\n[OK] System Ready!")

    while True:
        print_header("MAIN MENU")
        print("1. Find Similar Songs (by Song Name)")
        print("2. Recommend by Mood (Happy, Sad, etc.)")
        print("3. Recommend by Context (Party, Gym, etc.)")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            song_name = input("\nEnter song name: ").strip()
            artist_name = input("Enter artist name (optional, press Enter to skip): ").strip()
            if not artist_name: artist_name = None
            
            print(f"\nSearching for similar songs to '{song_name}'...")
            results = engine.get_similar_songs(song_name, artist_name, top_k=5)
            
            if not results:
                print("[X] Song not found or no recommendations generated.")
            else:
                print("\nTop 5 Recommendations:")
                print_separator()
                for i, res in enumerate(results, 1):
                    print(f"{i}. {format_song(res)}")
                print_separator()
                
        elif choice == '2':
            print("\nAvailable Moods: Happy, Sad, Energetic, Calm, Angry")
            mood = input("Enter mood: ").strip()
            
            print(f"\nGetting recommendations for '{mood}' mood...")
            results = engine.get_recommendations_by_mood(mood, top_k=10)
            
            if not results:
                print("[X] No songs found for this mood.")
            else:
                print(f"\nTop 10 Songs for {mood}:")
                print_separator()
                for i, res in enumerate(results, 1):
                    # Mood recommendation might not have similarity score
                    print(f"{i}. {res['song']} - {res['artist']}")
                    print(f"   Genre: {res['genre']} | Emotion: {res['emotion']}")
                print_separator()

        elif choice == '3':
            print("\nAvailable Contexts: Party, Workout, Study, Relax, Driving")
            context = input("Enter context: ").strip()
            
            print(f"\nGetting recommendations for '{context}'...")
            results = engine.get_recommendations_by_context(context, top_k=10)
            
            if not results:
                print("[X] No songs found for this context.")
            else:
                print(f"\nTop 10 Songs for {context}:")
                print_separator()
                for i, res in enumerate(results, 1):
                    print(f"{i}. {res['song']} - {res['artist']}")
                    print(f"   Genre: {res['genre']} | Emotion: {res['emotion']}")
                print_separator()
                
        elif choice == '4':
            print("\nGoodbye!")
            break
        else:
            print("\nInvalid choice. Please try again.")
            
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
