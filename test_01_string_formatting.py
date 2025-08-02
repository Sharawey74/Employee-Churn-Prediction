#!/usr/bin/env python3
"""
Test 1: Check string formatting issue with model results
"""

def test_string_formatting():
    """Test the specific formatting issue"""
    print("🧪 Testing string formatting issue...")
    
    # Simulate the problematic scenario
    model_results = {
        'best_score': 0.979,
        'training_time': 'N/A',  # This is a string, not a number
        'best_params': {}
    }
    
    try:
        # This will fail - trying to format string as float
        print(f"   Training time: {model_results.get('training_time', 'N/A'):.2f}s")
        print("❌ This should have failed!")
    except ValueError as e:
        print(f"✅ Caught expected error: {e}")
    
    # Fixed version - check type before formatting
    training_time = model_results.get('training_time', 'N/A')
    if isinstance(training_time, (int, float)):
        print(f"   Training time: {training_time:.2f}s")
    else:
        print(f"   Training time: {training_time}")
    
    print("✅ String formatting test completed!")

if __name__ == "__main__":
    test_string_formatting()
