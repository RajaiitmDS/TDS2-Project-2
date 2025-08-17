#!/usr/bin/env python3
"""
Test script for the Data Analyst Agent API
"""

import requests
import tempfile
import os

def test_basic_api():
    """Test the basic API functionality"""
    
    # Create a simple questions file
    questions_content = """
    Analyze the following questions:
    
    1. What is 2 + 2?
    2. What is the square root of 16?
    3. Create a simple test response.
    
    Return the answers as a JSON array.
    """
    
    # Create temporary questions file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(questions_content)
        questions_file = f.name
    
    try:
        # Test the API
        url = "http://localhost:5000/api/"
        
        with open(questions_file, 'rb') as f:
            files = {'questions.txt': f}
            
            print("Sending request to API...")
            response = requests.post(url, files=files, timeout=60)
            
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            
            if response.status_code == 200:
                print("✅ API test successful!")
                return True
            else:
                print("❌ API test failed!")
                return False
                
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False
        
    finally:
        # Clean up
        if os.path.exists(questions_file):
            os.unlink(questions_file)

def test_wikipedia_api():
    """Test the API with Wikipedia movie data"""
    
    questions_content = """
    Scrape the list of highest grossing films from Wikipedia. It is at the URL:
    https://en.wikipedia.org/wiki/List_of_highest-grossing_films

    Answer the following questions and respond with a JSON array of strings containing the answer.

    1. How many $2 bn movies were released before 2000?
    2. Which is the earliest film that grossed over $1.5 bn?
    3. What's the correlation between the Rank and Peak?
    4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
       Return as a base-64 encoded data URI, `"data:image/png;base64,iVBORw0KG..."` under 100,000 bytes.
    """
    
    # Create temporary questions file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(questions_content)
        questions_file = f.name
    
    try:
        # Test the API
        url = "http://localhost:5000/api/"
        
        with open(questions_file, 'rb') as f:
            files = {'questions.txt': f}
            
            print("Sending Wikipedia test request to API...")
            response = requests.post(url, files=files, timeout=180)  # Longer timeout for web scraping
            
            print(f"Status Code: {response.status_code}")
            print(f"Response length: {len(response.text)} characters")
            
            if response.status_code == 200:
                try:
                    import json
                    result = response.json()
                    print(f"✅ Wikipedia API test successful!")
                    print(f"Result type: {type(result)}")
                    if isinstance(result, list):
                        print(f"Array length: {len(result)}")
                        for i, item in enumerate(result):
                            if isinstance(item, str) and len(item) > 100:
                                print(f"Item {i}: {item[:100]}...")
                            else:
                                print(f"Item {i}: {item}")
                    return True
                except json.JSONDecodeError as e:
                    print(f"❌ Invalid JSON response: {e}")
                    print(f"Raw response: {response.text[:500]}...")
                    return False
            else:
                print("❌ Wikipedia API test failed!")
                print(f"Response: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Error testing Wikipedia API: {e}")
        return False
        
    finally:
        # Clean up
        if os.path.exists(questions_file):
            os.unlink(questions_file)

if __name__ == "__main__":
    print("Testing Data Analyst Agent API...")
    print("=" * 50)
    
    # Test basic functionality first
    print("\n1. Testing basic API functionality...")
    basic_success = test_basic_api()
    
    if basic_success:
        print("\n2. Testing Wikipedia data analysis...")
        wikipedia_success = test_wikipedia_api()
        
        if wikipedia_success:
            print("\n🎉 All tests passed! API is working correctly.")
        else:
            print("\n⚠️  Basic test passed but Wikipedia test failed.")
    else:
        print("\n❌ Basic API test failed. Check the application logs.")
    
    print("\n" + "=" * 50)