#!/usr/bin/env python3
"""
Test script that matches the exact evaluation format
"""

import requests
import tempfile
import os
import json

def test_evaluation_format():
    """Test the exact format expected by the evaluation system"""
    
    # Create the exact questions from the evaluation example
    questions_content = """Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings containing the answer.

1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
   Return as a base-64 encoded data URI, `"data:image/png;base64,iVBORw0KG..."` under 100,000 bytes."""
    
    # Create temporary questions file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(questions_content)
        questions_file = f.name
    
    try:
        # Test the API
        url = "http://localhost:5000/api/"
        
        with open(questions_file, 'rb') as f:
            files = {'questions.txt': f}
            
            print("🔬 Testing Evaluation Format...")
            print("=" * 60)
            
            response = requests.post(url, files=files, timeout=180)
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    
                    # Validate the response format
                    print(f"✅ Response received")
                    print(f"📊 Response type: {type(result)}")
                    
                    if isinstance(result, list):
                        print(f"📋 Array length: {len(result)}")
                        
                        # Validate each answer
                        if len(result) >= 4:
                            print("\n🔍 Analyzing answers:")
                            
                            # Answer 1: Should be an integer
                            print(f"Answer 1 (Movies before 2000): {result[0]} (type: {type(result[0])})")
                            if isinstance(result[0], int):
                                print("  ✅ Correct type (integer)")
                            else:
                                print("  ❌ Expected integer")
                            
                            # Answer 2: Should be a string with movie title
                            print(f"Answer 2 (Earliest $1.5bn film): {result[1]} (type: {type(result[1])})")
                            if isinstance(result[1], str) and len(result[1]) > 0:
                                print("  ✅ Correct type (string)")
                                if "titanic" in result[1].lower() or "avatar" in result[1].lower():
                                    print("  ✅ Reasonable movie title")
                                else:
                                    print(f"  ⚠️  Unexpected title: {result[1]}")
                            else:
                                print("  ❌ Expected non-empty string")
                            
                            # Answer 3: Should be a float (correlation)
                            print(f"Answer 3 (Correlation): {result[2]} (type: {type(result[2])})")
                            if isinstance(result[2], (int, float)):
                                print("  ✅ Correct type (numeric)")
                                if -1 <= result[2] <= 1:
                                    print("  ✅ Valid correlation range")
                                else:
                                    print("  ❌ Correlation out of range [-1, 1]")
                            else:
                                print("  ❌ Expected numeric value")
                            
                            # Answer 4: Should be a base64 data URI
                            print(f"Answer 4 (Plot): {str(result[3])[:50]}... (length: {len(str(result[3]))})")
                            if isinstance(result[3], str):
                                if result[3].startswith("data:image/png;base64,"):
                                    print("  ✅ Correct data URI format")
                                    if len(result[3]) < 100000:
                                        print("  ✅ Under 100KB size limit")
                                    else:
                                        print("  ❌ Exceeds 100KB size limit")
                                else:
                                    print("  ❌ Expected data:image/png;base64, prefix")
                            else:
                                print("  ❌ Expected string")
                            
                            print("\n📋 Final Response (evaluation format):")
                            print(json.dumps(result, indent=2))
                            
                            # Also show the exact format for curl
                            print("\n🌐 API Response Summary:")
                            print(f"[{result[0]}, \"{result[1]}\", {result[2]}, \"data:image/png;base64,...\"]")
                            
                            return True
                        else:
                            print(f"❌ Expected 4 answers, got {len(result)}")
                            return False
                    else:
                        print(f"❌ Expected array response, got {type(result)}")
                        return False
                        
                except json.JSONDecodeError as e:
                    print(f"❌ Invalid JSON response: {e}")
                    print(f"Raw response: {response.text[:500]}...")
                    return False
            else:
                print("❌ API request failed!")
                print(f"Response: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False
        
    finally:
        # Clean up
        if os.path.exists(questions_file):
            os.unlink(questions_file)

if __name__ == "__main__":
    print("🎯 Data Analyst Agent - Evaluation Format Test")
    print("=" * 60)
    
    success = test_evaluation_format()
    
    if success:
        print("\n🎉 EVALUATION TEST PASSED!")
        print("The API correctly implements the required format.")
        print("\n📤 Ready for deployment and evaluation!")
    else:
        print("\n❌ EVALUATION TEST FAILED!")
        print("The API response format needs adjustment.")
    
    print("\n" + "=" * 60)