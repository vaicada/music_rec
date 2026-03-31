"""
Image Mood Classifier - CLIP-based Image Analysis for Music Recommendation.

================================================================================
PURPOSE:
================================================================================
This module provides image analysis capabilities for the Music Recommender.
It uses OpenAI's CLIP model to classify images into mood/context categories,
which are then used to recommend suitable music.

================================================================================
ARCHITECTURE:
================================================================================
    User uploads image
           |
           v
    ImageMoodClassifier.analyze_image()
           |
           v
    CLIP Model (Zero-shot Classification)
           |
           v
    Returns (label, type) -> e.g., ("Happy", "mood")
           |
           v
    MusicRecommendationEngine.get_recommendations_by_mood/context()

================================================================================
HOW IT WORKS:
================================================================================
CLIP (Contrastive Language-Image Pre-training) can compare an image against
a set of text descriptions and determine which description best matches the
image. We leverage this by:

1. Defining "visual prompts" that describe emotional/contextual scenes
2. Mapping each prompt to a mood or context label in our music system
3. Running zero-shot classification to find the best match

Example:
    Image of a smiling person -> Matches "a photo of a happy smiling person"
    -> Maps to ("Happy", "mood") -> Returns happy songs

================================================================================
USAGE:
================================================================================
    from hybrid_music_engine.image_processor import ImageMoodClassifier
    
    classifier = ImageMoodClassifier()
    classifier.load_model()  # Optional: preload model at startup
    
    with open("happy_face.jpg", "rb") as f:
        image_bytes = f.read()
    
    label, label_type, confidence = classifier.analyze_image(image_bytes)
    # label = "Happy", label_type = "mood", confidence = 0.85

================================================================================
DEPENDENCIES:
================================================================================
- transformers (for CLIP model)
- pillow (for image processing)
- torch (for model inference)

================================================================================
Author: Graduation Project
Created: 2026-01-29
================================================================================
"""

import io
from typing import Tuple, Optional, Dict, List
from PIL import Image

# =============================================================================
# Global flag for CLIP availability
# =============================================================================
CLIP_AVAILABLE = False
try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
    CLIP_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] CLIP dependencies not available: {e}")
    print("[WARNING] Image-based recommendation feature will be disabled.")


class ImageMoodClassifier:
    """
    Classifies images into mood/context categories using CLIP.
    
    This class encapsulates all CLIP-related logic, keeping it separate from
    the main recommendation engine for cleaner code organization.
    
    Attributes:
        model_name (str): HuggingFace model identifier for CLIP
        model (CLIPModel): Loaded CLIP model instance
        processor (CLIPProcessor): CLIP image/text processor
        device (str): Device to run inference on ('cuda' or 'cpu')
        is_loaded (bool): Whether the model has been loaded
        
    Visual Prompt Mapping:
        Each prompt describes a visual scene and maps to a system label.
        The prompts are designed to cover common emotional and contextual
        scenarios that users might upload.
    """
    
    # =========================================================================
    # Class Constants: Visual Prompt Mapping
    # =========================================================================
    # Format: "visual description" -> (system_label, type)
    # type is either "mood" or "context" to match our recommendation API
    
    VISUAL_PROMPTS: Dict[str, Tuple[str, str]] = {
        # Mood-based prompts (for get_recommendations_by_mood)
        "a photo of a happy smiling person": ("Happy", "mood"),
        "a photo of a joyful celebration": ("Happy", "mood"),
        "a photo of a sad or crying person": ("Sad", "mood"),
        "a photo of a melancholic scene": ("Sad", "mood"),
        "a photo of an energetic concert or festival": ("Energetic", "mood"),
        "a photo of people jumping or dancing": ("Energetic", "mood"),
        "a peaceful sunset or sunrise landscape": ("Calm", "mood"),
        "a serene nature scene with trees or water": ("Calm", "mood"),
        "a photo of an angry or frustrated person": ("Angry", "mood"),
        
        # Context-based prompts (for get_recommendations_by_context)
        "a party scene with dancing people": ("Party", "context"),
        "a nightclub or disco with colorful lights": ("Party", "context"),
        "people working out in a gym": ("Workout", "context"),
        "a person running or jogging outdoors": ("Workout", "context"),
        "a quiet study room or library": ("Study", "context"),
        "a person reading or working at a desk": ("Study", "context"),
        "a relaxing spa or meditation scene": ("Relax", "context"),
        "a cozy living room or bedroom": ("Relax", "context"),
        "a scenic road trip or highway view": ("Driving", "context"),
        "a car interior or driving scene": ("Driving", "context"),
    }
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize the ImageMoodClassifier.
        
        Args:
            model_name: HuggingFace model identifier. Default is the base
                       CLIP model which offers a good balance of speed and
                       accuracy (~350MB download).
        
        Note:
            The model is NOT loaded during initialization to save memory.
            Call load_model() explicitly or it will be loaded on first use.
        """
        self.model_name = model_name
        self.model: Optional[CLIPModel] = None
        self.processor: Optional[CLIPProcessor] = None
        self.device = "cuda" if CLIP_AVAILABLE and torch.cuda.is_available() else "cpu"
        self.is_loaded = False
        
        # Pre-compute the list of text prompts for CLIP
        self._text_prompts: List[str] = list(self.VISUAL_PROMPTS.keys())
        
    def load_model(self) -> bool:
        """
        Load the CLIP model and processor.
        
        This method downloads the model from HuggingFace Hub on first call.
        Subsequent calls will use the cached model.
        
        Returns:
            bool: True if model loaded successfully, False otherwise.
            
        Raises:
            RuntimeError: If CLIP dependencies are not installed.
        """
        if not CLIP_AVAILABLE:
            print("[ERROR] Cannot load CLIP model: dependencies not installed.")
            print("[ERROR] Please run: pip install transformers pillow torch")
            return False
            
        if self.is_loaded:
            print("[INFO] CLIP model already loaded.")
            return True
            
        try:
            print(f"[INFO] Loading CLIP model: {self.model_name}")
            print(f"[INFO] This may take a moment on first run (downloading ~350MB)...")
            
            # Load processor and model
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model = CLIPModel.from_pretrained(self.model_name)
            
            # Move model to appropriate device
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            self.is_loaded = True
            print(f"[OK] CLIP model loaded successfully on {self.device.upper()}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to load CLIP model: {e}")
            self.is_loaded = False
            return False
    
    def analyze_image(self, image_bytes: bytes) -> Tuple[str, str, float]:
        """
        Analyze an image and return the detected mood/context.
        
        This is the main method for image classification. It:
        1. Loads the image from bytes
        2. Runs CLIP zero-shot classification against all visual prompts
        3. Returns the best matching mood/context label
        
        Args:
            image_bytes: Raw image data (JPEG, PNG, or WebP)
            
        Returns:
            Tuple of (label, type, confidence):
                - label: The detected mood or context (e.g., "Happy", "Workout")
                - type: Either "mood" or "context"
                - confidence: Float between 0 and 1 indicating match confidence
                
        Raises:
            ValueError: If image cannot be processed
            RuntimeError: If model is not loaded and cannot be loaded
            
        Example:
            >>> classifier = ImageMoodClassifier()
            >>> with open("sunset.jpg", "rb") as f:
            ...     label, ltype, conf = classifier.analyze_image(f.read())
            >>> print(f"Detected: {label} ({ltype}) with {conf:.1%} confidence")
            Detected: Calm (mood) with 85.3% confidence
        """
        # Ensure model is loaded
        if not self.is_loaded:
            if not self.load_model():
                raise RuntimeError("Failed to load CLIP model")
        
        # Load and validate image
        try:
            image = Image.open(io.BytesIO(image_bytes))
            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if image.mode != "RGB":
                image = image.convert("RGB")
        except Exception as e:
            raise ValueError(f"Cannot process image: {e}")
        
        # Prepare inputs for CLIP
        inputs = self.processor(
            text=self._text_prompts,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Get image-text similarity scores
            logits_per_image = outputs.logits_per_image  # Shape: [1, num_prompts]
            
            # Convert to probabilities using softmax
            probs = logits_per_image.softmax(dim=1)
            
            # Get the best matching prompt
            best_idx = probs.argmax().item()
            confidence = probs[0, best_idx].item()
            
        # Map the winning prompt to our mood/context label
        best_prompt = self._text_prompts[best_idx]
        label, label_type = self.VISUAL_PROMPTS[best_prompt]
        
        print(f"[INFO] Image analysis result: {label} ({label_type}) - {confidence:.1%}")
        
        return label, label_type, confidence
    
    def get_all_labels(self) -> Dict[str, List[str]]:
        """
        Get all available mood and context labels.
        
        Returns:
            Dictionary with 'moods' and 'contexts' keys, each containing
            a list of unique labels.
            
        Example:
            >>> classifier.get_all_labels()
            {'moods': ['Happy', 'Sad', 'Energetic', 'Calm', 'Angry'],
             'contexts': ['Party', 'Workout', 'Study', 'Relax', 'Driving']}
        """
        moods = set()
        contexts = set()
        
        for label, label_type in self.VISUAL_PROMPTS.values():
            if label_type == "mood":
                moods.add(label)
            else:
                contexts.add(label)
                
        return {
            "moods": sorted(list(moods)),
            "contexts": sorted(list(contexts))
        }


# =============================================================================
# Module-level convenience function
# =============================================================================

def is_clip_available() -> bool:
    """
    Check if CLIP dependencies are available.
    
    Returns:
        bool: True if transformers and torch are installed.
    """
    return CLIP_AVAILABLE
