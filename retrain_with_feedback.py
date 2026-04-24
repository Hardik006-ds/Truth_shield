# retrain_with_feedback.py
"""
Retrain model with feedback data to improve accuracy
"""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random

# Check if we have enough feedback to retrain
FEEDBACK_FILE = Path("feedback.json")

def load_feedback():
    if not FEEDBACK_FILE.exists():
        return {"correct": [], "incorrect": []}
    with open(FEEDBACK_FILE, 'r') as f:
        return json.load(f)

def create_feedback_dataset():
    """Create a mini dataset from feedback corrections"""
    feedback = load_feedback()
    incorrect = feedback.get("incorrect", [])
    
    if len(incorrect) < 5:
        print(f"\nNot enough feedback to retrain. Need at least 5 corrections, have {len(incorrect)}")
        return None
    
    print(f"\nFound {len(incorrect)} corrections to learn from!")
    
    # For now, just show the statistics
    print("\nFeedback Statistics:")
    print(f"  Correct predictions: {len(feedback.get('correct', []))}")
    print(f"  Incorrect predictions: {len(incorrect)}")
    
    # Analyze the incorrect predictions
    if incorrect:
        correction_types = {}
        for item in incorrect:
            verdict = item.get('predicted_verdict', 'unknown')
            correction_types[verdict] = correction_types.get(verdict, 0) + 1
        
        print("\n  Common mistakes:")
        for verdict, count in correction_types.items():
            print(f"    {verdict}: {count} times")
    
    return True

def show_feedback_stats():
    """Show current feedback statistics"""
    feedback = load_feedback()
    
    print("\n" + "="*50)
    print("FEEDBACK STATISTICS")
    print("="*50)
    print(f"Total correct feedback: {len(feedback.get('correct', []))}")
    print(f"Total incorrect feedback: {len(feedback.get('incorrect', []))}")
    
    if feedback.get('correct') or feedback.get('incorrect'):
        total = len(feedback.get('correct', [])) + len(feedback.get('incorrect', []))
        accuracy = len(feedback.get('correct', [])) / total * 100 if total > 0 else 0
        print(f"User feedback accuracy: {accuracy:.1f}%")
    
    print("="*50)
    
    # Check if ready to retrain
    if len(feedback.get('incorrect', [])) >= 5:
        print("\nReady to retrain! Run: python retrain_with_feedback.py --retrain")
    else:
        print(f"\nNeed {5 - len(feedback.get('incorrect', []))} more corrections to retrain")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--retrain":
        result = create_feedback_dataset()
        if result:
            print("\nRetraining with feedback data...")
            print("(This would retrain the model with corrected data)")
    else:
        show_feedback_stats()
