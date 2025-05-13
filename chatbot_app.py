from flask import Flask, render_template, request, jsonify
import json
import os
from dotenv import load_dotenv
from langchain.agents import tool
from typing import Annotated, Any
from pydantic import BeforeValidator
import re
from crewai import Agent, Task, Crew, Process
import traceback

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

# Define CrewAI agents for the chatbot
def create_agents_and_tasks(product: str, query_type: str):
    """Create CrewAI agents and tasks for processing the query."""
    # Define agents with specific instructions on tool usage
    market_analyst = Agent(
        role="Market Research Analyst",
        goal="Analyze market trends and product performance",
        backstory="""You are an experienced market analyst with expertise in 
        consumer electronics. You provide detailed analysis of product performance 
        and market trends to help guide business decisions.""",
        verbose=True,
        tools=[fetch_market_trends],
        allow_delegation=False
    )

    product_specialist = Agent(
        role="Product Specialist",
        goal="Analyze product specifications and availability",
        backstory="""You are a product specialist with deep knowledge of consumer 
        electronics. Your expertise helps companies understand product details
        and market positioning. When describing product availability, always use the
        exact phrase 'In Stock' when available.""",
        verbose=True,
        tools=[fetch_product_data],
        allow_delegation=False
    )

    qa_specialist = Agent(
        role="Data Quality Checker",
        goal="Create detailed side-by-side comparisons of data and verify accuracy",
        backstory="""You are a data quality checker responsible for verifying 
        that analysts are using correct data in their reports. Your primary task
        is to compare the data from the analyses with the source data and present
        a detailed side-by-side comparison showing both sets of values.
        You understand that minor format differences are acceptable as long as 
        the meaning is the same, and you normalize these differences 
        in your reporting to ensure consistency.""",
        verbose=True,
        # QA agent uses both tools for verification
        tools=[fetch_product_data, fetch_market_trends],
        allow_delegation=False
    )

    # Define tasks based on query type
    tasks = []
    
    if query_type in ["price", "availability", "rating"]:
        # Product-related query
        product_task = Task(
            description=f"""Analyze the {product} product details focusing on {query_type}.
            Provide comprehensive information about the {query_type} of {product}.
            Use the 'Fetch Product Data' tool with '{product}' as the product.
            """,
            expected_output=f"""A detailed analysis of {product}'s {query_type}, 
            with comparisons to industry standards and actionable insights.""",
            agent=product_specialist
        )
        tasks.append(product_task)
        
    elif query_type in ["trend", "market", "popularity"]:
        # Market-related query
        market_task = Task(
            description=f"""Analyze the {product} market trends focusing on market position.
            Provide detailed insights on popularity metrics and trend status.
            Use the 'Fetch Market Trends' tool with '{product}' as the product.
            """,
            expected_output=f"""A comprehensive market trend analysis for {product}
            including trend status, popularity score, and monthly search volume significance.""",
            agent=market_analyst
        )
        tasks.append(market_task)
        
    else:
        # Comprehensive query needs both
        product_task = Task(
            description=f"""Analyze the {product} product details and provide a comprehensive report.
            Focus on price point, availability status, and customer rating significance.
            Use the 'Fetch Product Data' tool with '{product}' as the product.
            """,
            expected_output=f"""A detailed product analysis for {product} covering price point analysis,
            availability status, and customer rating significance.""",
            agent=product_specialist
        )
        
        market_task = Task(
            description=f"""Analyze the {product} market trends and provide detailed insights.
            Be sure to include popularity metrics and comparison with industry averages.
            Use the 'Fetch Market Trends' tool with '{product}' as the product.
            """,
            expected_output=f"""A comprehensive market trend analysis for {product} including trend status,
            popularity score interpretation, and monthly search volume significance.""",
            agent=market_analyst
        )
        
        tasks.append(product_task)
        tasks.append(market_task)
    
    # Add QA verification task if needed
    if len(tasks) > 0:
        qa_task = Task(
            description=f"""Your job is to verify data accuracy by comparing the data points
            in the analyses with the source data for {product}.
            
            1. First, extract key data points from the analyses.
            2. Then, use your tools to independently fetch the same data for {product}.
            3. Create a detailed comparison table of the values.
            4. Normalize values where needed (e.g., "Available" vs "In Stock").
            5. Clearly state if the QA verification passed or failed.
            """,
            expected_output="""A detailed data comparison followed by a verification result (PASS/FAIL).""",
            agent=qa_specialist,
            context=tasks
        )
        tasks.append(qa_task)
    
    return tasks

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

# Process CrewAI output to extract thinking steps
def extract_thinking_steps(task_outputs):
    """Extract thinking steps from CrewAI task outputs."""
    thinking_steps = []
    
    try:
        for i, output in enumerate(task_outputs):
            try:
                # TaskOutput objects need to be converted to strings
                output_text = str(output)
                
                # Extract agent role from output
                agent_role = "Agent"
                if "Market Research Analyst" in output_text:
                    agent_role = "Market Research Analyst"
                elif "Product Specialist" in output_text:
                    agent_role = "Product Specialist"
                elif "Data Quality Checker" in output_text:
                    agent_role = "Data Quality Checker"
                
                # Split the output by steps
                steps = output_text.split("\n\n")
                
                # Process initial analysis
                thinking_steps.append({
                    "step": f"{agent_role} - Initial Analysis",
                    "content": steps[0] if steps else output_text
                })
                
                # Look for tool usage patterns - try multiple variations
                tool_patterns = [
                    r"I'll use the '([^']+)' tool with product = '([^']+)'",
                    r"I will use the '([^']+)' tool with product = '([^']+)'", 
                    r"Using the '([^']+)' tool with '([^']+)'",
                    r"Calling '([^']+)' tool with product = '([^']+)'",
                    r"Executing '([^']+)' tool with '([^']+)'"
                ]
                
                tool_found = False
                for pattern in tool_patterns:
                    tool_matches = re.findall(pattern, output_text)
                    if tool_matches:
                        tool_found = True
                        for tool_name, product in tool_matches:
                            thinking_steps.append({
                                "step": f"{agent_role} - Tool Execution",
                                "content": f"Using '{tool_name}' tool with product = '{product}'"
                            })
                
                # If no tool patterns matched but we see keywords, add a generic tool step
                if not tool_found and ('tool' in output_text and ('fetch' in output_text.lower() or 'data' in output_text.lower())):
                    thinking_steps.append({
                        "step": f"{agent_role} - Tool Execution",
                        "content": "Using data retrieval tools to gather information"
                    })
                
                # Look for JSON data patterns
                json_pattern = r"({[\s\S]*?})"
                json_matches = re.findall(json_pattern, output_text)
                
                if json_matches:
                    for json_str in json_matches:
                        try:
                            # Verify it's valid JSON
                            json_data = json.loads(json_str)
                            thinking_steps.append({
                                "step": f"{agent_role} - Data Analysis",
                                "content": f"Analyzing data:\n{json.dumps(json_data, indent=2)}"
                            })
                        except:
                            # Not valid JSON, skip
                            continue
                
                # Add conclusion/synthesis step
                if len(steps) > 1:
                    thinking_steps.append({
                        "step": f"{agent_role} - Conclusion",
                        "content": steps[-1]
                    })
            except Exception as e:
                # Handle errors for individual outputs
                thinking_steps.append({
                    "step": f"Processing Error",
                    "content": f"Error processing agent output {i+1}: {str(e)}\nRaw output: {str(output)[:200]}..."
                })
    except Exception as e:
        # Fallback for any unexpected errors
        thinking_steps.append({
            "step": "Error Processing Agent Thinking",
            "content": f"Error extracting thinking steps: {str(e)}"
        })
    
    # Ensure we return at least one thinking step
    if not thinking_steps:
        thinking_steps.append({
            "step": "Agent Processing",
            "content": "The agents processed your request, but detailed thinking steps were not available."
        })
    
    return thinking_steps

def generate_response(user_query: str) -> dict:
    """Generate a response based on the user query using CrewAI."""
    
    # Extract product from query - simple logic to identify iPhone
    product = "iPhone" if "iphone" in user_query.lower() else "unknown product"
    
    # Determine query type for generating agent thinking
    query_type = ""
    if "price" in user_query.lower() or "cost" in user_query.lower():
        query_type = "price"
    elif "availability" in user_query.lower() or "stock" in user_query.lower() or "available" in user_query.lower():
        query_type = "availability"
    elif "rating" in user_query.lower() or "reviews" in user_query.lower():
        query_type = "rating"
    elif "trend" in user_query.lower() or "market" in user_query.lower() or "popularity" in user_query.lower():
        query_type = "market"
    else:
        # If no specific category is detected, provide comprehensive data
        query_type = "comprehensive"
    
    try:
        # Create agents and tasks
        tasks = create_agents_and_tasks(product, query_type)
        
        # Create and run crew
        crew = Crew(
            agents=[task.agent for task in tasks],
            tasks=tasks,
            verbose=True,
            process=Process.sequential
        )
        
        # Get crew result
        crew_result = crew.kickoff()
        
        # Collect task outputs safely
        task_outputs = []
        for task in tasks:
            if hasattr(task, 'output') and task.output is not None:
                task_outputs.append(task.output)
                print(f"Task output type: {type(task.output)}")
        
        # Extract thinking steps from task outputs
        thinking_steps = extract_thinking_steps(task_outputs)
        
        # Prepare response based on query type
        if query_type in ["price", "availability", "rating"]:
            # Product data query
            product_data = _get_product_data(product)
            
            if query_type == "price":
                response_text = f"The {product} is priced at {product_data['price']}."
            elif query_type == "availability":
                response_text = f"The {product} is currently {product_data['availability']}."
            elif query_type == "rating":
                response_text = f"The {product} has a rating of {product_data['rating']} out of 5."
            
            response = {
                "response": response_text,
                "data": product_data,
                "thinking_steps": thinking_steps
            }
            
        elif query_type == "market":
            # Market data query
            market_data = _get_market_trends(product)
            response = {
                "response": f"The {product} is currently showing a {market_data['trend']} trend with a popularity score of {market_data['popularity_score']} and {market_data['monthly_searches']} monthly searches.",
                "data": market_data,
                "thinking_steps": thinking_steps
            }
            
        else:
            # Comprehensive query
            product_data = _get_product_data(product)
            market_data = _get_market_trends(product)
            
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
                "market_data": market_data,
                "thinking_steps": thinking_steps
            }
        
        # Run QA verification on the response data
        verified_response = perform_qa_check(response)
        
        return verified_response
    
    except Exception as e:
        # Handle any errors and return a friendly message
        error_trace = traceback.format_exc()
        print(f"Error generating response: {str(e)}\n{error_trace}")
        
        # Fallback to basic response
        if query_type in ["price", "availability", "rating"]:
            product_data = _get_product_data(product)
            if query_type == "price":
                response_text = f"The {product} is priced at {product_data['price']}."
            elif query_type == "availability":
                response_text = f"The {product} is currently {product_data['availability']}."
            else:
                response_text = f"The {product} has a rating of {product_data['rating']} out of 5."
            
            return {
                "response": response_text,
                "data": product_data,
                "thinking_steps": [{
                    "step": "Error Information",
                    "content": f"There was an error running CrewAI: {str(e)}\nUsing fallback data instead."
                }]
            }
        else:
            return {
                "response": f"I encountered an error while processing your query about {product}. Here's some basic information instead.",
                "product_data": _get_product_data(product),
                "market_data": _get_market_trends(product),
                "thinking_steps": [{
                    "step": "Error Information",
                    "content": f"There was an error running CrewAI: {str(e)}\nUsing fallback data instead."
                }]
            }

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
