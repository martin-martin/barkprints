#!/usr/bin/env python3
"""Example script demonstrating programmatic usage of barkprints."""

from barkprints.text_generator import TextGenerator

def main():
    # Create a generator instance
    generator = TextGenerator()
    
    # Generate a haiku from a bark image
    print("=== Nature Haiku ===")
    haiku = generator.generate("barks.jpg", "nature")
    print(haiku)
    
    # Generate news commentary from the same image
    print("\n=== News Commentary ===")
    commentary = generator.generate("barks.jpg", "news")
    print(commentary)
    
    # Use format override to create a sentence from nature vocabulary
    print("\n=== Nature Sentence ===")
    sentence = generator.generate("barks.jpg", "nature", "sentence")
    print(sentence)
    
    # The same image always produces the same output (deterministic)
    print("\n=== Verifying Determinism ===")
    haiku2 = generator.generate("barks.jpg", "nature")
    print(f"Haikus are identical: {haiku == haiku2}")

if __name__ == "__main__":
    main()

