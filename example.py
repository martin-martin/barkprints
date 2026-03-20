#!/usr/bin/env python3
"""Example script demonstrating programmatic usage of barkprints."""

from barkprints.text_generator import TextGenerator

def main():
    # Create a generator instance
    generator = TextGenerator()

    # Generate text from a bark image using nature corpus
    print("=== Nature Voice ===")
    nature_text = generator.generate("barks.jpg", "nature")
    print(nature_text)

    # Try with literature corpus
    print("\n=== Literature Voice ===")
    lit_text = generator.generate("barks.jpg", "literature")
    print(lit_text)

    # Adjust alpha: more coherent (0.2) vs more bark-driven (0.8)
    print("\n=== Alpha Variations ===")
    gen_coherent = TextGenerator(alpha=0.2)
    gen_bark = TextGenerator(alpha=0.8)
    print(f"Coherent (alpha=0.2): {gen_coherent.generate('barks.jpg', 'nature')}")
    print(f"Bark-driven (alpha=0.8): {gen_bark.generate('barks.jpg', 'nature')}")

    # The same image always produces the same output (deterministic)
    print("\n=== Verifying Determinism ===")
    nature_text2 = generator.generate("barks.jpg", "nature")
    print(f"Outputs are identical: {nature_text == nature_text2}")

    # Show how different corpora give different "voices" to the same bark
    print("\n=== Same Bark, Different Perspectives ===")
    print(f"Nature voice: {nature_text}")
    print(f"Literary voice: {lit_text}")
    print("\nThe bark's features steer a walk through each corpus's word space!")

if __name__ == "__main__":
    main()
