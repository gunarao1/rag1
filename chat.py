"""
My First Gen AI App - Chat with Google Gemini (FREE)
A simple chatbot to learn how LLMs work
"""

import google.generativeai as genai
import os

# Your API key - get from: https://aistudio.google.com/app/apikey
# Best practice: use environment variable
API_KEY = "AIzaSyDZxmiP5mumEAqR9YM0YFr87GAQ40X-KV0"  # Replace this OR set GOOGLE_API_KEY env variable

def main():
    """Main chat loop"""
    print("=== My First Gen AI App (Google Gemini - FREE) ===")
    print("Type your message (or 'quit' to exit)\n")
    
    # Configure Gemini with your API key
    genai.configure(api_key=API_KEY)
    
    # Initialize the model
    model = genai.GenerativeModel(
        model_name='gemini-2.5-flash',  # Fast and FREE
        system_instruction="You are a helpful AI assistant. Be concise and friendly."
    )
    
    # Start a chat session (handles conversation history automatically)
    chat = model.start_chat(history=[])
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'quit':
            break
            
        if not user_input:
            continue
        
        try:
            # Send message and get response
            response = chat.send_message(user_input)
            
            print(f"\nAI: {response.text}\n")
            
            # Optional: Show token usage (for learning)
            try:
                print(f"[Tokens - Input: {response.usage_metadata.prompt_token_count}, "
                      f"Output: {response.usage_metadata.candidates_token_count}]")
            except:
                pass  # Some responses don't include token counts
            
        except Exception as e:
            print(f"\nError: {e}\n")
    
    print("Goodbye!")


if __name__ == "__main__":
    main()


"""
=== HOW TO RUN THIS ===

STEP 1: INSTALL PYTHON
- Download from: python.org (Python 3.9+)
- Verify: python --version

STEP 2: INSTALL GOOGLE AI LIBRARY
pip install google-generativeai

STEP 3: GET FREE API KEY (NO CREDIT CARD!)
- Go to: https://aistudio.google.com/app/apikey
- Click "Create API Key"
- Copy the key
- Replace "your-api-key-here" above

STEP 4: RUN
python chat_app.py

=== BETTER: USE ENVIRONMENT VARIABLE ===
Instead of hardcoding API key:

# On Windows:
set GOOGLE_API_KEY=your-key-here
python chat_app.py

# On Mac/Linux:
export GOOGLE_API_KEY=your-key-here
python chat_app.py

Then change line 10 to:
API_KEY = os.getenv("GOOGLE_API_KEY")


=== KEY CONCEPTS EXPLAINED ===

1. CHAT SESSION:
   - chat = model.start_chat() creates a conversation
   - Gemini AUTOMATICALLY manages history for you!
   - No need to manually store messages (easier than Claude)

2. SYSTEM INSTRUCTION:
   - Sets AI personality/behavior
   - Try: "You are a Python expert" or "You are a pirate"
   - Changes how the AI responds

3. MODEL OPTIONS:
   - gemini-1.5-flash: Fast, FREE, good for learning
   - gemini-1.5-pro: Smarter but has rate limits
   - gemini-2.0-flash-exp: Newest experimental model

4. TOKEN USAGE:
   - Shows how much text was processed
   - Helps understand costs (even though it's free!)
   - Good habit for when you use paid APIs


=== WHY GEMINI IS GREAT FOR LEARNING ===

✅ COMPLETELY FREE (no credit card!)
✅ Generous rate limits (15 requests/minute)
✅ Good quality responses
✅ Simpler API (auto-handles conversation history)
✅ Multimodal (can handle images, video - advanced feature)


=== TRY THESE EXPERIMENTS ===

1. CHANGE PERSONALITY:
   system_instruction="You are a helpful Python tutor. Explain with code examples."

2. TRY DIFFERENT MODELS:
   model_name='gemini-1.5-pro'  # Smarter but slower
   model_name='gemini-2.0-flash-exp'  # Newest experimental

3. ADD TEMPERATURE (creativity):
   model = genai.GenerativeModel(
       model_name='gemini-1.5-flash',
       generation_config=genai.types.GenerationConfig(
           temperature=0.7,  # 0.0 = focused, 1.0 = creative
       )
   )

4. SET MAX OUTPUT LENGTH:
   generation_config=genai.types.GenerationConfig(
       max_output_tokens=500,  # Limit response length
   )

5. VIEW CONVERSATION HISTORY:
   # Add after chat.send_message():
   print(f"\n[History length: {len(chat.history)} messages]")


=== RATE LIMITS (FREE TIER) ===

Gemini 1.5 Flash (FREE):
- 15 requests per minute
- 1,500 requests per day
- 1 million tokens per minute

For learning = MORE than enough!


=== COMMON ERRORS & FIXES ===

1. "ModuleNotFoundError: No module named 'google.generativeai'"
   FIX: pip install google-generativeai

2. "Invalid API key"
   FIX: Get new key from aistudio.google.com/app/apikey

3. "Resource exhausted" (429 error)
   FIX: You hit rate limit, wait 1 minute

4. "Quota exceeded"
   FIX: Daily limit reached (rare), try tomorrow


=== GEMINI vs CLAUDE - COMPARISON ===

GEMINI ADVANTAGES:
✓ Completely FREE
✓ Auto-handles conversation history
✓ Multimodal (images, video)
✓ Faster responses

CLAUDE ADVANTAGES:
✓ Better at complex reasoning
✓ Longer context window (200K vs 1M tokens)
✓ Better code generation
✓ More reliable for production


=== WHAT'S HAPPENING UNDER THE HOOD ===

When you send a message:
1. Your text → Tokenized (split into pieces)
2. Previous conversation retrieved from chat session
3. Sent to Google's servers
4. Gemini processes everything and generates response
5. Response sent back
6. History automatically updated

The chat.send_message() does all this automatically!


=== COOL FEATURES TO TRY LATER ===

1. CHAT WITH IMAGES:
   from PIL import Image
   img = Image.open('photo.jpg')
   response = model.generate_content(['Describe this image:', img])

2. STREAMING RESPONSES (word-by-word):
   response = chat.send_message(user_input, stream=True)
   for chunk in response:
       print(chunk.text, end='', flush=True)

3. FUNCTION CALLING (let AI use tools):
   # Advanced feature - AI can call Python functions!


=== SWITCHING TO CLAUDE LATER ===

When you get Claude credits, just change:

FROM:
import google.generativeai as genai
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')
chat = model.start_chat()
response = chat.send_message(user_input)

TO:
import anthropic
client = anthropic.Anthropic(api_key=API_KEY)
# (use the previous Claude code)

Both do the same thing, just different syntax!


=== NEXT STEPS TO LEARN ===

Once this works:

LEVEL 2 - Add Features:
□ Streaming responses (word-by-word)
□ Save conversations to file
□ Add command to clear history
□ Count total tokens used

LEVEL 3 - Build Something Real:
□ Web interface (Flask/Streamlit)
□ Chat with PDFs (RAG)
□ AI that can use tools (function calling)

LEVEL 4 - Advanced:
□ Vector databases (semantic search)
□ AI agents
□ Deploy to cloud

Ready to start? Just get your free API key and run it!
"""