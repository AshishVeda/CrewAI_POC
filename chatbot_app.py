from flask import Flask, render_template, request, jsonify
import json
import os
from dotenv import load_dotenv
from langchain.agents import tool
from typing import Annotated, Any
from pydantic import BeforeValidator
import re

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Helper function to extract product name from various input formats
def parse_product_input(value: Any) -> str:
    """Extract product name from various input formats."""
    if isinstance(value, dict):
        # Handle dictionary formats
        if 'product' in value and isinstance(value['product'], str):
            return value['product']
    elif isinstance(value, str):
        # Handle string inputs
        return value
    # Return value as is for other cases
    return str(value)

# Define the annotated type with validator
ProductInput = Annotated[str, BeforeValidator(parse_product_input)]

# Define tools using the simpler @tool decorator with type annotations
@tool("Fetch Product Data")
def fetch_product_data(product: ProductInput) -> str:
    """Fetch product pricing, availability, and rating for a given product name."""
    try:
        print(f"Fetching product data for: {product}")
        data = _get_product_data(product)
        return json.dumps(data)
    except Exception as e:
        return f"Error fetching product data: {str(e)}"

@tool("Fetch Market Trends")
def fetch_market_trends(product: ProductInput) -> str:
    """Fetch trend status and popularity metrics for the given product name."""
    try:
        print(f"Fetching market trends for: {product}")
        data = _get_market_trends(product)
        return json.dumps(data)
    except Exception as e:
        return f"Error fetching market trends: {str(e)}"

def _get_product_data(product: str) -> dict:
    """Fetch product data from a simulated database."""
    if product.lower() == "iphone":
        return {
            "product": "iPhone",
            "price": "$999",
            "availability": "In Stock",
            "rating": 4.8
        }
    return {
        "product": product, 
        "price": "$199", 
        "availability": "Out of Stock",
        "rating": 3.5
    }

def _get_market_trends(product: str) -> dict:
    """Fetch market trends from a simulated database."""
    if product.lower() == "iphone":
        return {
            "product": "iPhone",
            "trend": "Rising",
            "popularity_score": 92,
            "monthly_searches": 45000
        }
    return {
        "product": product,
        "trend": "Stable",
        "popularity_score": 70,
        "monthly_searches": 12000
    }

# Function to simulate agent thinking process
def generate_agent_thinking(product: str, query_type: str) -> list:
    """Generate simulated agent thinking process steps."""
    thinking_steps = []
    
    # Initial thinking
    thinking_steps.append({
        "step": "Initial Analysis",
        "content": f"I need to analyze the user's query about {product}. Based on the query, I should focus on {query_type} information."
    })
    
    # Data retrieval thinking
    if query_type in ["price", "availability", "rating"]:
        thinking_steps.append({
            "step": "Data Retrieval Plan",
            "content": f"I'll need to fetch product data for {product} to get accurate {query_type} information."
        })
        
        thinking_steps.append({
            "step": "Executing Tool",
            "content": f"Calling 'Fetch Product Data' tool with product = '{product}'"
        })
        
        # Simulated data analysis
        data = _get_product_data(product)
        thinking_steps.append({
            "step": "Data Analysis",
            "content": f"Received product data: {json.dumps(data, indent=2)}\n\nThe {query_type} information is: {data.get(query_type, 'Not found')}"
        })
    
    elif query_type in ["trend", "market", "popularity"]:
        thinking_steps.append({
            "step": "Data Retrieval Plan",
            "content": f"I'll need to fetch market trend data for {product} to understand its market position and popularity."
        })
        
        thinking_steps.append({
            "step": "Executing Tool",
            "content": f"Calling 'Fetch Market Trends' tool with product = '{product}'"
        })
        
        # Simulated market analysis
        data = _get_market_trends(product)
        thinking_steps.append({
            "step": "Data Analysis",
            "content": f"Received market data: {json.dumps(data, indent=2)}\n\nAnalyzing key points:\n- Trend status: {data['trend']}\n- Popularity score: {data['popularity_score']}/100\n- Monthly searches: {data['monthly_searches']}"
        })
        
        # Add industry comparison thinking
        thinking_steps.append({
            "step": "Industry Comparison",
            "content": "Comparing with industry averages:\n- Average smartphone popularity score: 75/100\n- Average monthly searches: 25,000\n\nThis indicates that the iPhone is performing above average in the market."
        })
    
    else:
        # Comprehensive analysis thinking
        thinking_steps.append({
            "step": "Comprehensive Analysis Plan",
            "content": f"I need to retrieve both product specifications and market trend data for {product} to provide a complete overview."
        })
        
        thinking_steps.append({
            "step": "Executing Product Data Tool",
            "content": f"Calling 'Fetch Product Data' tool with product = '{product}'"
        })
        
        # Product data analysis
        product_data = _get_product_data(product)
        thinking_steps.append({
            "step": "Product Data Analysis",
            "content": f"Received product data: {json.dumps(product_data, indent=2)}"
        })
        
        thinking_steps.append({
            "step": "Executing Market Trends Tool",
            "content": f"Calling 'Fetch Market Trends' tool with product = '{product}'"
        })
        
        # Market data analysis
        market_data = _get_market_trends(product)
        thinking_steps.append({
            "step": "Market Data Analysis",
            "content": f"Received market data: {json.dumps(market_data, indent=2)}"
        })
        
        # Combined analysis
        thinking_steps.append({
            "step": "Synthesizing Information",
            "content": f"Combining product and market data to generate a comprehensive overview of {product}.\n\n"
                      f"Key points:\n"
                      f"- Premium price point of {product_data['price']}\n"
                      f"- Customer satisfaction rating of {product_data['rating']}/5\n"
                      f"- {market_data['trend']} market trend with popularity score of {market_data['popularity_score']}/100\n"
                      f"- High monthly search volume of {market_data['monthly_searches']}\n\n"
                      f"These factors together indicate that {product} is a popular, premium product with strong market presence."
        })
    
    # Final response formulation
    thinking_steps.append({
        "step": "Response Formulation",
        "content": "Formulating a clear and concise response based on the analyzed data."
    })
    
    return thinking_steps

# QA verification functions
def normalize_availability_status(status):
    """Normalize availability status strings to match regardless of exact wording."""
    if not status:
        return ""
    status = status.lower().strip()
    if "stock" in status or "available" in status:
        return "In Stock"
    return status

def compare_values(a, b):
    """Compare two values with normalization for certain data types."""
    # Convert both to strings
    str_a = str(a).strip()
    str_b = str(b).strip()
    
    # Check if they're already equal
    if str_a == str_b:
        return True
    
    # Normalize ratings
    if "/" in str_a:
        str_a = str_a.split("/")[0]
    if "/" in str_b:
        str_b = str_b.split("/")[0]
    
    # Normalize numbers (remove commas)
    str_a = str_a.replace(",", "")
    str_b = str_b.replace(",", "")
    
    # Normalize availability
    if any(keyword in str_a.lower() for keyword in ["available", "in stock", "stock"]):
        str_a = "In Stock"
    if any(keyword in str_b.lower() for keyword in ["available", "in stock", "stock"]):
        str_b = "In Stock"
    
    return str_a == str_b

def perform_qa_check(response_data):
    """Performs QA verification on the response data"""
    # Extract the data from the response
    if 'data' in response_data:
        # Single data source response
        reported_data = response_data['data']
        
        # Get product name
        product_name = reported_data.get('product', 'iPhone')
        
        # Directly fetch the data again for verification
        actual_data = None
        qa_result = {
            "passed": True,
            "comparison": [],
            "message": "QA verification passed"
        }
        
        # Determine which data type we're dealing with and fetch the reference data
        if 'popularity_score' in reported_data:
            # It's market trends data
            actual_data = _get_market_trends(product_name)
            data_type = "market_trends"
            
            # Compare each field
            for key in ['trend', 'popularity_score', 'monthly_searches']:
                if key in reported_data and key in actual_data:
                    matches = compare_values(reported_data[key], actual_data[key])
                    qa_result["comparison"].append({
                        "field": key,
                        "reported": reported_data[key],
                        "actual": actual_data[key],
                        "matches": matches
                    })
                    if not matches:
                        qa_result["passed"] = False
                        qa_result["message"] = f"QA failed: Discrepancy found in {key}"
        
        elif 'price' in reported_data:
            # It's product data
            actual_data = _get_product_data(product_name)
            data_type = "product_data"
            
            # Compare each field
            for key in ['price', 'availability', 'rating']:
                if key in reported_data and key in actual_data:
                    matches = compare_values(reported_data[key], actual_data[key])
                    qa_result["comparison"].append({
                        "field": key,
                        "reported": reported_data[key],
                        "actual": actual_data[key],
                        "matches": matches
                    })
                    if not matches:
                        qa_result["passed"] = False
                        qa_result["message"] = f"QA failed: Discrepancy found in {key}"
        
        # Add QA results to the response
        response_data['qa_result'] = qa_result
        
    elif 'product_data' in response_data and 'market_data' in response_data:
        # Combined data response
        product_data = response_data['product_data']
        market_data = response_data['market_data']
        
        # Get product name
        product_name = product_data.get('product', 'iPhone')
        
        # Fetch actual data for verification
        actual_product_data = _get_product_data(product_name)
        actual_market_data = _get_market_trends(product_name)
        
        qa_result = {
            "passed": True,
            "product_comparison": [],
            "market_comparison": [],
            "message": "QA verification passed"
        }
        
        # Check product data
        for key in ['price', 'availability', 'rating']:
            if key in product_data and key in actual_product_data:
                matches = compare_values(product_data[key], actual_product_data[key])
                qa_result["product_comparison"].append({
                    "field": key,
                    "reported": product_data[key],
                    "actual": actual_product_data[key],
                    "matches": matches
                })
                if not matches:
                    qa_result["passed"] = False
                    qa_result["message"] = f"QA failed: Discrepancy found in product {key}"
        
        # Check market data
        for key in ['trend', 'popularity_score', 'monthly_searches']:
            if key in market_data and key in actual_market_data:
                matches = compare_values(market_data[key], actual_market_data[key])
                qa_result["market_comparison"].append({
                    "field": key,
                    "reported": market_data[key],
                    "actual": actual_market_data[key],
                    "matches": matches
                })
                if not matches:
                    qa_result["passed"] = False
                    qa_result["message"] = f"QA failed: Discrepancy found in market {key}"
        
        # Add QA results to the response
        response_data['qa_result'] = qa_result
    
    return response_data

def generate_response(user_query: str) -> dict:
    """Generate a response based on the user query."""
    
    # Extract product from query - simple logic to identify iPhone
    product = "iPhone" if "iphone" in user_query.lower() else "unknown product"
    
    # Determine query type for generating agent thinking
    query_type = ""
    if "price" in user_query.lower() or "cost" in user_query.lower():
        query_type = "price"
        product_data = json.loads(fetch_product_data(product))
        response = {
            "response": f"The {product} is priced at {product_data['price']}.",
            "data": product_data
        }
    elif "availability" in user_query.lower() or "stock" in user_query.lower() or "available" in user_query.lower():
        query_type = "availability"
        product_data = json.loads(fetch_product_data(product))
        response = {
            "response": f"The {product} is currently {product_data['availability']}.",
            "data": product_data
        }
    elif "rating" in user_query.lower() or "reviews" in user_query.lower():
        query_type = "rating"
        product_data = json.loads(fetch_product_data(product))
        response = {
            "response": f"The {product} has a rating of {product_data['rating']} out of 5.",
            "data": product_data
        }
    elif "trend" in user_query.lower() or "market" in user_query.lower() or "popularity" in user_query.lower():
        query_type = "market"
        market_data = json.loads(fetch_market_trends(product))
        response = {
            "response": f"The {product} is currently showing a {market_data['trend']} trend with a popularity score of {market_data['popularity_score']} and {market_data['monthly_searches']} monthly searches.",
            "data": market_data
        }
    else:
        # If no specific category is detected, provide both product and market data
        query_type = "comprehensive"
        product_data = json.loads(fetch_product_data(product))
        market_data = json.loads(fetch_market_trends(product))
        
        response = {
            "response": f"Here's what I found about the {product}:\n\n" + 
                       f"Price: {product_data['price']}\n" +
                       f"Availability: {product_data['availability']}\n" +
                       f"Rating: {product_data['rating']} out of 5\n\n" +
                       f"Market Trends:\n" +
                       f"Trend: {market_data['trend']}\n" +
                       f"Popularity Score: {market_data['popularity_score']}\n" +
                       f"Monthly Searches: {market_data['monthly_searches']}",
            "product_data": product_data,
            "market_data": market_data
        }
    
    # Generate agent thinking process
    response['thinking_steps'] = generate_agent_thinking(product, query_type)
    
    # Run QA verification on the response data
    verified_response = perform_qa_check(response)
    
    return verified_response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    
    # Generate response based on user message
    response_data = generate_response(user_message)
    
    return jsonify(response_data)

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    app.run(debug=True) 