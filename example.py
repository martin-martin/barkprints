#!/usr/bin/env python3
"""Example script demonstrating programmatic usage of barkprints."""

from barkprints.text_generator import TextGenerator

def main():
    # Create a generator instance
    generator = TextGenerator()
    
    # Generate text from a bark image using nature corpus
    print("=== Nature Match ===")
    nature_text = generator.generate("barks.jpg", "nature")
    print(nature_text)
    
    # Try with literature corpus
    print("\n=== Literature Match ===")
    lit_text = generator.generate("barks.jpg", "literature")
    print(lit_text)
    
    # Get top 3 matches with similarity scores
    print("\n=== Top 3 Nature Matches ===")
    matches = generator.generate("barks.jpg", "nature", top_k=3)
    for i, (sentence, score) in enumerate(matches, 1):
        print(f"{i}. [{score:.3f}] {sentence}")
    
    # The same image always produces the same output (deterministic)
    print("\n=== Verifying Determinism ===")
    nature_text2 = generator.generate("barks.jpg", "nature")
    print(f"Outputs are identical: {nature_text == nature_text2}")
    
    # Show how different corpora give different "voices" to the same bark
    print("\n=== Same Bark, Different Perspectives ===")
    print(f"Nature voice: {nature_text}")
    print(f"Literary voice: {lit_text}")
    print("\nThe bark's features map to different positions in different semantic spaces!")

if __name__ == "__main__":
    main()
