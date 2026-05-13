#!/usr/bin/env python3
"""
Test script for Kannada Text Normalization

This script tests the cardinal number normalization for Kannada.
Run with: python test_kannada_tn.py
"""

import sys
sys.path.insert(0, '.')

import pynini

from nemo_text_processing.text_normalization.kn.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.kn.taggers.decimal import DecimalFst
from nemo_text_processing.text_normalization.kn.taggers.money import MoneyFst
from nemo_text_processing.text_normalization.kn.verbalizers.cardinal import CardinalFst as CardinalVerbalizerFst
from nemo_text_processing.text_normalization.kn.taggers.tokenize_and_classify import ClassifyFst
from nemo_text_processing.text_normalization.kn.verbalizers.verbalize_final import VerbalizeFinalFst


def apply_fst(text: str, fst: pynini.Fst) -> str:
    """Apply FST to input text and return the output."""
    try:
        output = pynini.shortestpath(pynini.compose(text, fst)).string()
        return output
    except pynini.FstOpError:
        return f"[ERROR: No valid transduction for '{text}']"


def test_cardinal_tagger():
    """Test the cardinal tagger independently."""
    print("=" * 60)
    print("Testing Cardinal Tagger (Classifier)")
    print("=" * 60)
    
    cardinal = CardinalFst(deterministic=True)
    
    test_cases = [
        # Single digits - Kannada numerals
        ("೧", "cardinal { integer: \"ಒಂದು\" }"),
        ("೫", "cardinal { integer: \"ಐದು\" }"),
        ("೦", "cardinal { integer: \"ಸೊನ್ನೆ\" }"),
        # Single digits - Arabic numerals
        ("1", "cardinal { integer: \"ಒಂದು\" }"),
        ("5", "cardinal { integer: \"ಐದು\" }"),
        ("0", "cardinal { integer: \"ಸೊನ್ನೆ\" }"),
        # Teens and ties - Kannada
        ("೧೦", "cardinal { integer: \"ಹತ್ತು\" }"),
        ("೨೩", "cardinal { integer: \"ಇಪ್ಪತ್ತಮೂರು\" }"),
        ("೯೯", "cardinal { integer: \"ತೊಂಬತ್ತೊಂಬತ್ತು\" }"),
        # Teens and ties - Arabic
        ("10", "cardinal { integer: \"ಹತ್ತು\" }"),
        ("23", "cardinal { integer: \"ಇಪ್ಪತ್ತಮೂರು\" }"),
        ("99", "cardinal { integer: \"ತೊಂಬತ್ತೊಂಬತ್ತು\" }"),
        # Hundreds (using contracted Kannada forms)
        # Standalone: 500 -> ಐನೂರು (with ು)
        # With digits: 523 -> ಐನೂರ ಇಪ್ಪತ್ತಮೂರು (without ು)
        ("100", "cardinal { integer: \"ನೂರು\" }"),
        ("123", "cardinal { integer: \"ನೂರ ಇಪ್ಪತ್ತಮೂರು\" }"),
        ("೧೨೩", "cardinal { integer: \"ನೂರ ಇಪ್ಪತ್ತಮೂರು\" }"),
        ("500", "cardinal { integer: \"ಐನೂರು\" }"),
        ("523", "cardinal { integer: \"ಐನೂರ ಇಪ್ಪತ್ತಮೂರು\" }"),
        # Thousands - with -ದ connector when followed by remainder
        ("1000", "cardinal { integer: \"ಒಂದು ಸಾವಿರ\" }"),
        ("1001", "cardinal { integer: \"ಒಂದು ಸಾವಿರದ ಒಂದು\" }"),
        ("5432", "cardinal { integer: \"ಐದು ಸಾವಿರದ ನಾನೂರ ಮೂವತ್ತೆರಡು\" }"),
        # Lakhs - with -ದ connector
        ("100000", "cardinal { integer: \"ಒಂದು ಲಕ್ಷ\" }"),
        ("123456", "cardinal { integer: \"ಒಂದು ಲಕ್ಷದ ಇಪ್ಪತ್ತಮೂರು ಸಾವಿರದ ನಾನೂರ ಐವತ್ತಾರು\" }"),
        # Crores - with -ಯ connector (vowel ending)
        ("10000000", "cardinal { integer: \"ಒಂದು ಕೋಟಿ\" }"),
        ("10000001", "cardinal { integer: \"ಒಂದು ಕೋಟಿಯ ಒಂದು\" }"),
        # Negative numbers
        ("-5", "cardinal { negative: \"true\" integer: \"ಐದು\" }"),
    ]
    
    passed = 0
    failed = 0
    
    for input_text, expected in test_cases:
        result = apply_fst(input_text, cardinal.fst)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"{status} Input: {input_text:15} -> {result}")
        if result != expected:
            print(f"  Expected: {expected}")
    
    print(f"\nResults: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    return failed == 0


def test_cardinal_verbalizer():
    """Test the cardinal verbalizer independently."""
    print("\n" + "=" * 60)
    print("Testing Cardinal Verbalizer")
    print("=" * 60)
    
    verbalizer = CardinalVerbalizerFst(deterministic=True)
    
    test_cases = [
        ('cardinal { integer: "ಒಂದು" }', "ಒಂದು"),
        ('cardinal { integer: "ಹತ್ತು" }', "ಹತ್ತು"),
        ('cardinal { integer: "ಇಪ್ಪತ್ತಮೂರು" }', "ಇಪ್ಪತ್ತಮೂರು"),
        ('cardinal { integer: "ಒಂದು ನೂರು" }', "ಒಂದು ನೂರು"),
        ('cardinal { negative: "true" integer: "ಐದು" }', "ಮೈನಸ್ ಐದು"),
    ]
    
    passed = 0
    failed = 0
    
    for input_text, expected in test_cases:
        result = apply_fst(input_text, verbalizer.fst)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"{status} Input: {input_text}")
        print(f"   Output: {result}")
        if result != expected:
            print(f"   Expected: {expected}")
    
    print(f"\nResults: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    return failed == 0


def test_full_pipeline():
    """Test the full TN pipeline (classify + verbalize)."""
    print("\n" + "=" * 60)
    print("Testing Full Pipeline (Classify -> Verbalize)")
    print("=" * 60)
    
    classifier = ClassifyFst(deterministic=True)
    verbalizer = VerbalizeFinalFst(deterministic=True)
    
    test_cases = [
        ("123", "ನೂರ ಇಪ್ಪತ್ತಮೂರು"),
        ("೧೨೩", "ನೂರ ಇಪ್ಪತ್ತಮೂರು"),
        ("5", "ಐದು"),
        ("೫", "ಐದು"),
        ("500", "ಐನೂರು"),
        ("523", "ಐನೂರ ಇಪ್ಪತ್ತಮೂರು"),
        ("1000", "ಒಂದು ಸಾವಿರ"),
        ("99", "ತೊಂಬತ್ತೊಂಬತ್ತು"),
    ]
    
    passed = 0
    failed = 0
    
    for input_text, expected in test_cases:
        # First classify
        classified = apply_fst(input_text, classifier.fst)
        # Then verbalize
        result = apply_fst(classified, verbalizer.fst)
        
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"{status} Input: {input_text:15} -> {result}")
        if result != expected:
            print(f"  Classified: {classified}")
            print(f"  Expected: {expected}")
    
    print(f"\nResults: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    return failed == 0


def test_regression_punctuation():
    """Regression test: numbers followed by punctuation."""
    print("\n" + "=" * 60)
    print("Regression Test: Punctuation Handling")
    print("=" * 60)
    
    classifier = ClassifyFst(deterministic=True)
    verbalizer = VerbalizeFinalFst(deterministic=True)
    
    test_cases = [
        ("5.", "ಐದು ."),
        ("100.", "ನೂರು ."),
        ("5,", "ಐದು ,"),
        ("123!", "ನೂರ ಇಪ್ಪತ್ತಮೂರು !"),
        ("500.", "ಐನೂರು ."),
        ("523!", "ಐನೂರ ಇಪ್ಪತ್ತಮೂರು !"),
    ]
    
    passed = 0
    failed = 0
    
    for input_text, expected in test_cases:
        classified = apply_fst(input_text, classifier.fst)
        result = apply_fst(classified, verbalizer.fst)
        
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"{status} Input: {input_text:15} -> {result}")
        if result != expected:
            print(f"  Classified: {classified}")
            print(f"  Expected: {expected}")
    
    print(f"\nResults: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    return failed == 0


def test_regression_comma_numbers():
    """Regression test: comma-grouped numbers (Western and Indian formats)."""
    print("\n" + "=" * 60)
    print("Regression Test: Comma-Grouped Numbers")
    print("=" * 60)
    
    cardinal = CardinalFst(deterministic=True)
    
    test_cases = [
        # Western format
        ("1,000", "cardinal { integer: \"ಒಂದು ಸಾವಿರ\" }"),
        ("1,000,000", "cardinal { integer: \"ಹತ್ತು ಲಕ್ಷ\" }"),
        # Indian format (12,34,567 = 12 lakh 34 thousand 567) - with -ದ connectors
        ("1,00,000", "cardinal { integer: \"ಒಂದು ಲಕ್ಷ\" }"),
        ("12,34,567", "cardinal { integer: \"ಹನ್ನೆರಡು ಲಕ್ಷದ ಮೂವತ್ತನಾಲ್ಕು ಸಾವಿರದ ಐನೂರ ಅರವತ್ತೇಳು\" }"),
        ("1,00,00,000", "cardinal { integer: \"ಒಂದು ಕೋಟಿ\" }"),
    ]
    
    passed = 0
    failed = 0
    
    for input_text, expected in test_cases:
        result = apply_fst(input_text, cardinal.fst)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"{status} Input: {input_text:15} -> {result}")
        if result != expected:
            print(f"  Expected: {expected}")
    
    print(f"\nResults: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    return failed == 0


def test_regression_large_numbers():
    """Regression test: large numbers using natural ಕೋಟಿ-based forms."""
    print("\n" + "=" * 60)
    print("Regression Test: Large Numbers (ನೂರು ಕೋಟಿ, ಸಾವಿರ ಕೋಟಿ)")
    print("=" * 60)
    
    cardinal = CardinalFst(deterministic=True)
    
    test_cases = [
        # 100 crore (10^9) - uses ನೂರು ಕೋಟಿ instead of ಅರಬ್
        ("1000000000", "cardinal { integer: \"ನೂರು ಕೋಟಿ\" }"),
        ("5000000000", "cardinal { integer: \"ಐನೂರು ಕೋಟಿ\" }"),
        # 1000 crore (10^10)
        ("10000000000", "cardinal { integer: \"ಒಂದು ಸಾವಿರ ಕೋಟಿ\" }"),
        # 10000 crore (10^11) - uses ಹತ್ತು ಸಾವಿರ ಕೋಟಿ instead of ಖರಬ್
        ("100000000000", "cardinal { integer: \"ಹತ್ತು ಸಾವಿರ ಕೋಟಿ\" }"),
        # 1 lakh crore (10^12)
        ("1000000000000", "cardinal { integer: \"ಒಂದು ಲಕ್ಷ ಕೋಟಿ\" }"),
    ]
    
    passed = 0
    failed = 0
    
    for input_text, expected in test_cases:
        result = apply_fst(input_text, cardinal.fst)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"{status} Input: {input_text:20} -> {result}")
        if result != expected:
            print(f"  Expected: {expected}")
    
    print(f"\nResults: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    return failed == 0


def test_money():
    """Test money normalization."""
    print("\n" + "=" * 60)
    print("Testing Money Normalization")
    print("=" * 60)
    
    classifier = ClassifyFst(deterministic=True)
    verbalizer = VerbalizeFinalFst(deterministic=True)
    
    test_cases = [
        # Rupees
        ("₹50", "ಐವತ್ತು ರೂಪಾಯಿ"),
        ("₹100", "ನೂರು ರೂಪಾಯಿ"),
        ("₹1000", "ಒಂದು ಸಾವಿರ ರೂಪಾಯಿ"),
        # Dollars
        ("$100", "ನೂರು ಡಾಲರ್"),
        ("$25", "ಇಪ್ಪತ್ತೈದು ಡಾಲರ್"),
        # Euros
        ("€50", "ಐವತ್ತು ಯೂರೋ"),
        # Fractional money - leading zero handled correctly
        ("₹0.05", "ಐದು ಪೈಸೆ"),  # 5 paise (not "zero five")
        ("₹1.05", "ಒಂದು ರೂಪಾಯಿ ಐದು ಪೈಸೆ"),  # 1 rupee 5 paise
        # Single-digit fraction interpreted as X0
        ("₹1.5", "ಒಂದು ರೂಪಾಯಿ ಐವತ್ತು ಪೈಸೆ"),  # 1 rupee 50 paise
        ("₹10.5", "ಹತ್ತು ರೂಪಾಯಿ ಐವತ್ತು ಪೈಸೆ"),  # 10 rupees 50 paise
        # Normal fractional money
        ("₹50.50", "ಐವತ್ತು ರೂಪಾಯಿ ಐವತ್ತು ಪೈಸೆ"),
    ]
    
    passed = 0
    failed = 0
    
    for input_text, expected in test_cases:
        classified = apply_fst(input_text, classifier.fst)
        result = apply_fst(classified, verbalizer.fst)
        
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"{status} Input: {input_text:15} -> {result}")
        if result != expected:
            print(f"  Classified: {classified}")
            print(f"  Expected: {expected}")
    
    print(f"\nResults: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    return failed == 0


def test_decimal():
    """Test decimal number normalization."""
    print("\n" + "=" * 60)
    print("Testing Decimal Normalization")
    print("=" * 60)
    
    classifier = ClassifyFst(deterministic=True)
    verbalizer = VerbalizeFinalFst(deterministic=True)
    
    test_cases = [
        # Basic decimals
        ("3.14", "ಮೂರು ದಶಮಾಂಶ ಒಂದು ನಾಲ್ಕು"),
        ("0.5", "ಸೊನ್ನೆ ದಶಮಾಂಶ ಐದು"),
        ("12.005", "ಹನ್ನೆರಡು ದಶಮಾಂಶ ಸೊನ್ನೆ ಸೊನ್ನೆ ಐದು"),
        # Negative decimals (no leading space)
        ("-3.14", "ಮೈನಸ್ ಮೂರು ದಶಮಾಂಶ ಒಂದು ನಾಲ್ಕು"),
        # Kannada numerals
        ("೩.೧೪", "ಮೂರು ದಶಮಾಂಶ ಒಂದು ನಾಲ್ಕು"),
        # Decimal with quantity (space and no-space)
        ("1.5ಲಕ್ಷ", "ಒಂದು ದಶಮಾಂಶ ಐದು ಲಕ್ಷ"),
        ("1.5 ಲಕ್ಷ", "ಒಂದು ದಶಮಾಂಶ ಐದು ಲಕ್ಷ"),
    ]
    
    passed = 0
    failed = 0
    
    for input_text, expected in test_cases:
        classified = apply_fst(input_text, classifier.fst)
        result = apply_fst(classified, verbalizer.fst)
        
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"{status} Input: {input_text:15} -> {result}")
        if result != expected:
            print(f"  Classified: {classified}")
            print(f"  Expected: {expected}")
    
    print(f"\nResults: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    return failed == 0


def test_ordinal():
    """Test ordinal number normalization."""
    print("\n" + "=" * 60)
    print("Testing Ordinal Normalization")
    print("=" * 60)
    
    classifier = ClassifyFst(deterministic=True)
    verbalizer = VerbalizeFinalFst(deterministic=True)
    
    test_cases = [
        # Kannada ordinal suffix ನೇ - basic
        ("1ನೇ", "ಮೊದಲನೆಯ"),
        ("2ನೇ", "ಎರಡನೆಯ"),
        ("3ನೇ", "ಮೂರನೆಯ"),
        ("10ನೇ", "ಹತ್ತನೆಯ"),
        ("20ನೇ", "ಇಪ್ಪತ್ತನೆಯ"),
        # Full coverage - previously unsupported
        ("22ನೇ", "ಇಪ್ಪತ್ತೆರಡನೆಯ"),
        ("25ನೇ", "ಇಪ್ಪತ್ತೈದನೆಯ"),
        ("57ನೇ", "ಐವತ್ತೇಳನೆಯ"),
        ("99ನೇ", "ತೊಂಬತ್ತೊಂಬತ್ತನೆಯ"),
        # Hundreds
        ("100ನೇ", "ನೂರನೆಯ"),
        ("101ನೇ", "ನೂರ ಒಂದನೆಯ"),
        ("500ನೇ", "ಐನೂರನೆಯ"),
        # Kannada numerals
        ("೧ನೇ", "ಮೊದಲನೆಯ"),
        ("೧೦ನೇ", "ಹತ್ತನೆಯ"),
        ("೨೨ನೇ", "ಇಪ್ಪತ್ತೆರಡನೆಯ"),
        # English ordinal suffixes
        ("1st", "ಮೊದಲನೆಯ"),
        ("2nd", "ಎರಡನೆಯ"),
        ("3rd", "ಮೂರನೆಯ"),
        ("4th", "ನಾಲ್ಕನೆಯ"),
        ("10th", "ಹತ್ತನೆಯ"),
        ("22nd", "ಇಪ್ಪತ್ತೆರಡನೆಯ"),
    ]
    
    passed = 0
    failed = 0
    
    for input_text, expected in test_cases:
        classified = apply_fst(input_text, classifier.fst)
        result = apply_fst(classified, verbalizer.fst)
        
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"{status} Input: {input_text:15} -> {result}")
        if result != expected:
            print(f"  Classified: {classified}")
            print(f"  Expected: {expected}")
    
    print(f"\nResults: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    return failed == 0


def test_telephone():
    """Test telephone number normalization with context-based detection."""
    print("\n" + "=" * 60)
    print("Testing Telephone Normalization (Context-Based)")
    print("=" * 60)
    
    classifier = ClassifyFst(deterministic=True)
    verbalizer = VerbalizeFinalFst(deterministic=True)
    
    test_cases = [
        # Mobile with context keyword (keyword preserved, no leading zero if not in input)
        ("ಮೊಬೈಲ್ 9876543210", "ಮೊಬೈಲ್ ಒಂಬತ್ತು ಎಂಟು ಏಳು ಆರು ಐದು ನಾಲ್ಕು ಮೂರು ಎರಡು ಒಂದು ಸೊನ್ನೆ"),
        # Mobile with leading 0 in input (zero preserved)
        ("ಮೊಬೈಲ್ 09876543210", "ಮೊಬೈಲ್ ಸೊನ್ನೆ ಒಂಬತ್ತು ಎಂಟು ಏಳು ಆರು ಐದು ನಾಲ್ಕು ಮೂರು ಎರಡು ಒಂದು ಸೊನ್ನೆ"),
        # Phone with context window (both keyword and window preserved)
        ("ಫೋನ್ ನಂಬರ್ 9876543210", "ಫೋನ್ ನಂಬರ್ ಒಂಬತ್ತು ಎಂಟು ಏಳು ಆರು ಐದು ನಾಲ್ಕು ಮೂರು ಎರಡು ಒಂದು ಸೊನ್ನೆ"),
        # Phone with country code (no context needed)
        ("+91 9876543210", "ಪ್ಲಸ್ ಒಂಬತ್ತು ಒಂದು  ಒಂಬತ್ತು ಎಂಟು ಏಳು ಆರು ಐದು ನಾಲ್ಕು ಮೂರು ಎರಡು ಒಂದು ಸೊನ್ನೆ"),
        # Pincode with context (keyword preserved)
        ("ಪಿನ್‌ಕೋಡ್ 560001", "ಪಿನ್‌ಕೋಡ್ ಐದು ಆರು ಸೊನ್ನೆ ಸೊನ್ನೆ ಸೊನ್ನೆ ಒಂದು"),
        # Without context -> treated as cardinal (not telephone)
        ("9876543210", "ಒಂಬೈನೂರ ಎಂಬತ್ತೇಳು ಕೋಟಿಯ ಅರವತ್ತೈದು ಲಕ್ಷದ ನಲವತ್ತಮೂರು ಸಾವಿರದ ಇನ್ನೂರ ಹತ್ತು"),
        ("560001", "ಐದು ಲಕ್ಷದ ಅರವತ್ತು ಸಾವಿರದ ಒಂದು"),
        # Extension support (11 digits = 10 + 1 extension)
        ("ಮೊಬೈಲ್ 98765432101", "ಮೊಬೈಲ್ ಒಂಬತ್ತು ಎಂಟು ಏಳು ಆರು ಐದು ನಾಲ್ಕು ಮೂರು ಎರಡು ಒಂದು ಸೊನ್ನೆ ಒಂದು "),
    ]
    
    passed = 0
    failed = 0
    
    for input_text, expected in test_cases:
        classified = apply_fst(input_text, classifier.fst)
        result = apply_fst(classified, verbalizer.fst)
        
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"{status} Input: {input_text:30} -> {result}")
        if result != expected:
            print(f"  Classified: {classified}")
            print(f"  Expected: {expected}")
    
    print(f"\nResults: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    return failed == 0


def main():
    print("Kannada Text Normalization - Test Suite")
    print("=" * 60)
    print()
    
    all_passed = True
    
    try:
        if not test_cardinal_tagger():
            all_passed = False
    except Exception as e:
        print(f"Cardinal tagger test failed with error: {e}")
        all_passed = False
    
    try:
        if not test_cardinal_verbalizer():
            all_passed = False
    except Exception as e:
        print(f"Cardinal verbalizer test failed with error: {e}")
        all_passed = False
    
    try:
        if not test_full_pipeline():
            all_passed = False
    except Exception as e:
        print(f"Full pipeline test failed with error: {e}")
        all_passed = False
    
    # Regression tests
    try:
        if not test_regression_punctuation():
            all_passed = False
    except Exception as e:
        print(f"Punctuation regression test failed with error: {e}")
        all_passed = False
    
    try:
        if not test_regression_comma_numbers():
            all_passed = False
    except Exception as e:
        print(f"Comma numbers regression test failed with error: {e}")
        all_passed = False
    
    try:
        if not test_regression_large_numbers():
            all_passed = False
    except Exception as e:
        print(f"Large numbers regression test failed with error: {e}")
        all_passed = False
    
    # New tests for Money and Decimal
    try:
        if not test_money():
            all_passed = False
    except Exception as e:
        print(f"Money test failed with error: {e}")
        all_passed = False
    
    try:
        if not test_decimal():
            all_passed = False
    except Exception as e:
        print(f"Decimal test failed with error: {e}")
        all_passed = False
    
    # Ordinal and Telephone tests
    try:
        if not test_ordinal():
            all_passed = False
    except Exception as e:
        print(f"Ordinal test failed with error: {e}")
        all_passed = False
    
    try:
        if not test_telephone():
            all_passed = False
    except Exception as e:
        print(f"Telephone test failed with error: {e}")
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! ✓")
    else:
        print("Some tests failed. Please review the output above.")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
