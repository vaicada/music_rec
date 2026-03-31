"""
Example Usage Script for the Hybrid Music Recommendation Engine.

This script demonstrates how to use the complete pipeline:
1. Load and preprocess data
2. Train the hybrid model
3. Build the FAISS index
4. Get recommendations

Author: Graduation Project
Created: 2026-01-06
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hybrid_music_engine import (
    Config,
    get_config,
    HybridMusicModel,
    DataManager,
    Trainer,
    MusicRecommendationEngine,
    get_logger
)


def train_model(args):
    """Train the hybrid model."""
    print("=" * 60)
    print("HYBRID MUSIC RECOMMENDATION ENGINE - TRAINING")
    print("=" * 60)
    
    # Initialize config
    config = get_config()
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.device = args.device
    
    # Initialize logger
    logger = get_logger(args.log_file)
    logger.log_section("Training Session Started")
    
    # Initialize data manager
    print("\n[1/5] Loading data...")
    data_manager = DataManager(config)
    data = data_manager.load_data(args.data_path)
    
    # Preprocess
    print("[2/5] Preprocessing data...")
    data = data_manager.preprocess(data, fit_processors=True)
    
    # Split data
    print("[3/5] Splitting data...")
    train_data, val_data, test_data = data_manager.split_data(
        data, 
        val_size=0.1, 
        test_size=0.1
    )
    
    # Create model
    print("[4/5] Creating model...")
    model = HybridMusicModel(config)
    
    # Create trainer
    trainer = Trainer(model, data_manager, config)
    
    # Train
    print("[5/5] Training model...")
    history = trainer.train(train_data, val_data, epochs=args.epochs)
    
    # Save final model
    model_path = os.path.join(config.paths.model_dir, "best_model.pth")
    os.makedirs(config.paths.model_dir, exist_ok=True)
    model.save(model_path)
    
    print("\n" + "=" * 60)
    print(f"Training completed! Model saved to: {model_path}")
    print("=" * 60)
    
    return model, data_manager


def build_index(args):
    """Build the FAISS index."""
    print("=" * 60)
    print("HYBRID MUSIC RECOMMENDATION ENGINE - BUILD INDEX")
    print("=" * 60)
    
    config = get_config()
    
    # Initialize engine
    engine = MusicRecommendationEngine(config)
    
    # Load model
    print("\n[1/2] Loading model...")
    engine.load_model(args.model_path)
    
    # Build index
    print("[2/2] Building FAISS index...")
    engine.build_index(args.data_path, args.output_path)
    
    print("\n" + "=" * 60)
    print(f"Index built and saved to: {args.output_path}")
    print("=" * 60)


def recommend(args):
    """Get song recommendations."""
    print("=" * 60)
    print("HYBRID MUSIC RECOMMENDATION ENGINE - RECOMMENDATIONS")
    print("=" * 60)
    
    config = get_config()
    
    # Initialize engine
    engine = MusicRecommendationEngine(config)
    
    # Load components
    print("\n[1/3] Loading model...")
    engine.load_model(args.model_path)
    
    print("[2/3] Loading FAISS index...")
    engine.load_index(args.index_path)
    
    print("[3/3] Loading song database...")
    engine.load_song_data(args.data_path)
    
    # Get recommendations
    print(f"\n{'=' * 60}")
    print(f"Finding songs similar to: {args.song}")
    if args.artist:
        print(f"By artist: {args.artist}")
    print(f"{'=' * 60}\n")
    
    recommendations = engine.get_similar_songs(
        song_name=args.song,
        artist_name=args.artist,
        top_k=args.top_k
    )
    
    if not recommendations:
        print("No recommendations found. Song may not be in the database.")
        return
    
    print(f"Top {len(recommendations)} recommendations:\n")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['song']}")
        print(f"   Artist: {rec['artist']}")
        print(f"   Similarity: {rec['similarity']:.2%}")
        if rec.get('genre'):
            print(f"   Genre: {rec['genre']}")
        if rec.get('emotion'):
            print(f"   Emotion: {rec['emotion']}")
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Hybrid Music Recommendation Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  python example_usage.py train --data data/songs.csv --epochs 10

  # Build FAISS index
  python example_usage.py index --model models/best_model.pth --data data/songs.csv

  # Get recommendations
  python example_usage.py recommend --song "Bohemian Rhapsody" --artist "Queen"
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data", dest="data_path", required=True,
                             help="Path to training data")
    train_parser.add_argument("--epochs", type=int, default=10,
                             help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=32,
                             help="Batch size")
    train_parser.add_argument("--device", default="cuda",
                             help="Device (cuda/cpu)")
    train_parser.add_argument("--log-file", default="project_implementation_log.txt",
                             help="Log file path")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Build FAISS index")
    index_parser.add_argument("--model", dest="model_path", required=True,
                             help="Path to trained model")
    index_parser.add_argument("--data", dest="data_path", required=True,
                             help="Path to song data")
    index_parser.add_argument("--output", dest="output_path",
                             default="models/faiss_index.bin",
                             help="Output path for index")
    
    # Recommend command
    rec_parser = subparsers.add_parser("recommend", help="Get recommendations")
    rec_parser.add_argument("--song", required=True,
                           help="Song name to find similar songs for")
    rec_parser.add_argument("--artist", default=None,
                           help="Artist name (for disambiguation)")
    rec_parser.add_argument("--top-k", type=int, default=5,
                           help="Number of recommendations")
    rec_parser.add_argument("--model", dest="model_path",
                           default="models/best_model.pth",
                           help="Path to trained model")
    rec_parser.add_argument("--index", dest="index_path",
                           default="models/faiss_index.bin",
                           help="Path to FAISS index")
    rec_parser.add_argument("--data", dest="data_path",
                           default="data/songs.csv",
                           help="Path to song database")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_model(args)
    elif args.command == "index":
        build_index(args)
    elif args.command == "recommend":
        recommend(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
