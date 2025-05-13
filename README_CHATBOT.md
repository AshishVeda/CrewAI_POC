# iPhone Product Info Chatbot

A simple web-based chatbot interface that provides information about iPhone products using CrewAI tools.

## Features

- Ask about iPhone prices, availability, ratings, and market trends
- Simple, clean web interface
- Suggested query buttons for quick access to common questions
- Real-time responses with formatted data display

## Installation

1. Make sure you have Python 3.8+ installed on your system.

2. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

3. Set up a virtual environment (recommended):
   ```
   python -m venv chatbot_env
   source chatbot_env/bin/activate  # On Windows: chatbot_env\Scripts\activate
   ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

5. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Running the Chatbot

1. Start the Flask application:
   ```
   python chatbot_app.py
   ```

2. Open your web browser and go to:
   ```
   http://127.0.0.1:5000/
   ```

3. You should see the chatbot interface where you can ask questions about the iPhone.

## Sample Questions

Here are some questions you can ask the chatbot:

- What is the price of the iPhone?
- Is the iPhone available?
- What is the rating of the iPhone?
- What are the market trends for iPhone?
- Tell me everything about the iPhone

## How It Works

The chatbot uses Flask for the web server and CrewAI tools to fetch product data. The tools are implemented as simple functions with the `@tool` decorator from CrewAI.

When you ask a question:
1. The web interface sends your question to the Flask backend
2. The backend analyzes your question and calls the appropriate tools
3. The tools retrieve the requested information (simulated data in this demo)
4. The response is formatted and sent back to the web interface
5. The web interface displays the response with formatted data

## Customization

You can modify the `_get_product_data` and `_get_market_trends` functions in `chatbot_app.py` to fetch real data from APIs or databases instead of using the simulated data.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 