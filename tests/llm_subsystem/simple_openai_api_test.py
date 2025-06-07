#!/usr/bin/env python3
"""
Simple OpenAI API connectivity test to verify setup.
"""
import os
import httpx
import json

def test_openai_api():
    """Test direct OpenAI API connectivity."""
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-api-key-here'")
        return False
    
    print(f"✅ API key found: {api_key[:8]}...")
    
    # Test the correct OpenAI endpoint
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-4o-mini",  # Using the correct model name
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello, this is a test.' and nothing else."}
        ],
        "max_tokens": 50,
        "temperature": 0.1
    }
    
    print(f"🔗 Testing endpoint: {url}")
    print(f"📦 Using model: {payload['model']}")
    
    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(url, json=payload, headers=headers)
            
            print(f"📡 Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                print(f"✅ Success! Response: {content}")
                print(f"📊 Tokens used: {result.get('usage', {})}")
                return True
            else:
                print(f"❌ API Error: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"Error details: {json.dumps(error_data, indent=2)}")
                except:
                    print(f"Raw response: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing OpenAI API connectivity...")
    success = test_openai_api()
    exit(0 if success else 1)