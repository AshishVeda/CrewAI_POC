from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain.tools import tool
from typing import Dict, List

# Define custom tools
@tool("Fetch Product Data")
def fetch_product_data(product: str) -> Dict:
    """Fetch product pricing, availability, and rating for a given product name."""
    if product.lower() == "iphone":
        return {
            "product": "iPhone",
            "price": "$999",
            "availability": "In Stock",
            "rating": 4.8
        }
    elif product.lower() == "samsung galaxy":
        return {
            "product": "Samsung Galaxy",
            "price": "$899",
            "availability": "In Stock",
            "rating": 4.6
        }
    return {
        "product": product, 
        "price": "$199", 
        "availability": "Out of Stock",
        "rating": 3.5
    }


@tool("Fetch Market Trends")
def fetch_market_trends(product: str) -> Dict:
    """Fetch trend status and popularity metrics for the given product name."""
    if product.lower() == "iphone":
        return {
            "product": "iPhone",
            "trend": "Rising",
            "popularity_score": 92,
            "monthly_searches": 45000
        }
    elif product.lower() == "samsung galaxy":
        return {
            "product": "Samsung Galaxy",
            "trend": "Stable",
            "popularity_score": 85,
            "monthly_searches": 38000
        }
    return {
        "product": product,
        "trend": "Stable",
        "popularity_score": 70,
        "monthly_searches": 12000
    }


@tool("Get Competitor Analysis")
def get_competitor_analysis(product: str) -> Dict:
    """Get competitive analysis data for a specific product."""
    if product.lower() == "iphone":
        return {
            "product": "iPhone",
            "main_competitors": ["Samsung Galaxy", "Google Pixel", "Xiaomi"],
            "market_share": "23%",
            "competitive_advantage": "Brand loyalty and ecosystem integration"
        }
    elif product.lower() == "samsung galaxy":
        return {
            "product": "Samsung Galaxy",
            "main_competitors": ["iPhone", "Google Pixel", "OnePlus"],
            "market_share": "19%",
            "competitive_advantage": "Hardware innovation and display technology"
        }
    return {
        "product": product,
        "main_competitors": ["Various brands"],
        "market_share": "5%",
        "competitive_advantage": "Price point"
    }


@tool("Get Customer Feedback")
def get_customer_feedback(product: str) -> Dict:
    """Get summarized customer feedback for a specific product."""
    if product.lower() == "iphone":
        return {
            "product": "iPhone",
            "positive_points": ["Camera quality", "Performance", "Ecosystem"],
            "negative_points": ["Battery life", "Price", "Charging speed"],
            "common_issues": ["Screen durability", "Storage limitations"],
            "satisfaction_score": 87
        }
    elif product.lower() == "samsung galaxy":
        return {
            "product": "Samsung Galaxy",
            "positive_points": ["Display quality", "Customization", "Camera versatility"],
            "negative_points": ["Software updates", "Bloatware"],
            "common_issues": ["Battery degradation", "Overheating during gaming"],
            "satisfaction_score": 83
        }
    return {
        "product": product,
        "positive_points": ["Affordable", "Basic functionality"],
        "negative_points": ["Performance", "Build quality"],
        "common_issues": ["Short lifespan", "Limited support"],
        "satisfaction_score": 65
    }


# Define agents
market_analyst = Agent(
    role="Market Research Analyst",
    goal="Analyze market trends and provide strategic insights",
    backstory="""You are an experienced market analyst with expertise in 
    consumer electronics. You provide detailed analysis of product performance 
    and market trends to help guide business decisions.""",
    verbose=True,
    tools=[fetch_market_trends, get_competitor_analysis]
)

product_specialist = Agent(
    role="Product Specialist",
    goal="Analyze product specifications and consumer demand",
    backstory="""You are a product specialist with deep knowledge of consumer 
    electronics. Your expertise helps companies understand product details
    and market positioning.""",
    verbose=True,
    tools=[fetch_product_data, get_customer_feedback]
)

marketing_strategist = Agent(
    role="Marketing Strategist",
    goal="Develop effective marketing strategies based on market and product data",
    backstory="""You are a marketing expert who specializes in creating 
    data-driven marketing strategies. You understand how to position products 
    in competitive markets and highlight key selling points.""",
    verbose=True,
    tools=[get_competitor_analysis, get_customer_feedback]
)

business_advisor = Agent(
    role="Business Strategy Advisor",
    goal="Synthesize insights and provide actionable business recommendations",
    backstory="""You are a seasoned business consultant who helps companies make 
    strategic decisions. You excel at integrating various data points and analyses 
    to form coherent business strategies.""",
    verbose=True,
    tools=[]  # This agent will rely on the outputs from other agents
)

# Define tasks
market_analysis_task = Task(
    description="""Analyze the smartphone market trends with a focus on iPhone and Samsung Galaxy.
    Be sure to include:
    1. Current trend status for both products
    2. Popularity score interpretation
    3. Monthly search volume comparison
    4. Competitive landscape analysis
    Your output will be used by the business advisor to form recommendations.
    """,
    agent=market_analyst
)

product_analysis_task = Task(
    description="""Analyze the iPhone and Samsung Galaxy product details and customer feedback.
    Focus on:
    1. Price point comparison
    2. Availability status
    3. Customer rating and satisfaction scores
    4. Key positive and negative feedback points
    Compare these products and identify their strengths and weaknesses.
    """,
    agent=product_specialist
)

marketing_strategy_task = Task(
    description="""Develop marketing strategy recommendations for a smartphone manufacturer
    looking to compete with iPhone and Samsung Galaxy. Use competitor analysis and customer 
    feedback to identify:
    1. Key differentiators to emphasize
    2. Target audience segments
    3. Positioning strategy
    4. Marketing message priorities
    Your strategies should be data-driven and actionable.
    """,
    agent=marketing_strategist,
    context=[market_analysis_task, product_analysis_task]
)

business_recommendation_task = Task(
    description="""Based on the market analysis, product analysis, and marketing strategy,
    develop comprehensive business recommendations for a smartphone manufacturer. Include:
    1. Product development priorities
    2. Market positioning strategy
    3. Competitive strategy
    4. Key investment areas
    5. Risk assessment and mitigation strategies
    Your recommendations should be specific, actionable, and backed by the data provided.
    """,
    agent=business_advisor,
    context=[market_analysis_task, product_analysis_task, marketing_strategy_task]
)

# Create crew
smartphone_market_crew = Crew(
    agents=[market_analyst, product_specialist, marketing_strategist, business_advisor],
    tasks=[market_analysis_task, product_analysis_task, marketing_strategy_task, business_recommendation_task],
    verbose=2,
    process=Process.sequential
)

# Execute crew
result = smartphone_market_crew.kickoff()
print("\n==== Advanced CrewAI POC Results ====\n")
print(result) 