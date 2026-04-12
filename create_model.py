"""
Script to generate placeholder classifier model.
Run this to create the classifier.pkl file.
"""

import pickle
import os
from pathlib import Path


def create_placeholder_model():
    """Create a placeholder trained model."""
    # Simple dictionary-based placeholder
    model = {
        'type': 'classifier',
        'status': 'placeholder',
        'message': 'Replace with actual trained model'
    }
    
    # Ensure models directory exists
    Path('models').mkdir(exist_ok=True)
    
    # Save model
    with open('models/classifier.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Placeholder classifier.pkl created successfully!")


if __name__ == "__main__":
    create_placeholder_model()
