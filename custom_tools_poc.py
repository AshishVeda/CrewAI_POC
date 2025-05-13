from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from typing import Dict, List, Optional
import json

# Custom tool classes
class ProductDataTool(BaseTool):
    name: str = "Fetch Product Data"
    description: str = "Fetch product pricing, availability, and rating for a given product name."
    
    def _run(self, product: str) -> str:
        """Run the tool."""
        data = self._fetch_product_data(product)
        return json.dumps(data)
    
    def _fetch_product_data(self, product: str) -> Dict:
        """Fetch product data from a simulated database."""
        product_db = {
            "iphone": {
                "product": "iPhone",
                "price": "$999",
                "availability": "In Stock",
                "rating": 4.8
            },
            "samsung galaxy": {
                "product": "Samsung Galaxy",
                "price": "$899",
                "availability": "In Stock",
                "rating": 4.6
            },
            "google pixel": {
                "product": "Google Pixel",
                "price": "$799",
                "availability": "Limited Stock",
                "rating": 4.5
            }
        }
        
        return product_db.get(product.lower(), {
            "product": product,
            "price": "$199",
            "availability": "Out of Stock",
            "rating": 3.5
        })


class MarketTrendsTool(BaseTool):
    name: str = "Fetch Market Trends"
    description: str = "Fetch trend status and popularity metrics for the given product name."
    
    def _run(self, product: str) -> str:
        """Run the tool."""
        data = self._fetch_market_trends(product)
        return json.dumps(data)
    
    def _fetch_market_trends(self, product: str) -> Dict:
        """Fetch market trends from a simulated database."""
        trends_db = {
            "iphone": {
                "product": "iPhone",
                "trend": "Rising",
                "popularity_score": 92,
                "monthly_searches": 45000
            },
            "samsung galaxy": {
                "product": "Samsung Galaxy",
                "trend": "Stable",
                "popularity_score": 85,
                "monthly_searches": 38000
            },
            "google pixel": {
                "product": "Google Pixel",
                "trend": "Rising",
                "popularity_score": 78,
                "monthly_searches": 25000
            }
        }
        
        return trends_db.get(product.lower(), {
            "product": product,
            "trend": "Stable",
            "popularity_score": 70,
            "monthly_searches": 12000
        })


class CompetitorAnalysisTool(BaseTool):
    name: str = "Get Competitor Analysis"
    description: str = "Get competitive analysis data for a specific product."
    
    def _run(self, product: str) -> str:
        """Run the tool."""
        data = self._get_competitor_analysis(product)
        return json.dumps(data)
    
    def _get_competitor_analysis(self, product: str) -> Dict:
        """Get competitor analysis from a simulated database."""
        competitor_db = {
            "iphone": {
                "product": "iPhone",
                "main_competitors": ["Samsung Galaxy", "Google Pixel", "Xiaomi"],
                "market_share": "23%",
                "competitive_advantage": "Brand loyalty and ecosystem integration"
            },
            "samsung galaxy": {
                "product": "Samsung Galaxy",
                "main_competitors": ["iPhone", "Google Pixel", "OnePlus"],
                "market_share": "19%",
                "competitive_advantage": "Hardware innovation and display technology"
            },
            "google pixel": {
                "product": "Google Pixel",
                "main_competitors": ["iPhone", "Samsung Galaxy", "OnePlus"],
                "market_share": "8%",
                "competitive_advantage": "Camera technology and software experience"
            }
        }
        
        return competitor_db.get(product.lower(), {
            "product": product,
            "main_competitors": ["Various brands"],
            "market_share": "5%",
            "competitive_advantage": "Price point"
        })


class CustomerFeedbackTool(BaseTool):
    name: str = "Get Customer Feedback"
    description: str = "Get summarized customer feedback for a specific product."
    
    def _run(self, product: str) -> str:
        """Run the tool."""
        data = self._get_customer_feedback(product)
        return json.dumps(data)
    
    def _get_customer_feedback(self, product: str) -> Dict:
        """Get customer feedback from a simulated database."""
        feedback_db = {
            "iphone": {
                "product": "iPhone",
                "positive_points": ["Camera quality", "Performance", "Ecosystem"],
                "negative_points": ["Battery life", "Price", "Charging speed"],
                "common_issues": ["Screen durability", "Storage limitations"],
                "satisfaction_score": 87
            },
            "samsung galaxy": {
                "product": "Samsung Galaxy",
                "positive_points": ["Display quality", "Customization", "Camera versatility"],
                "negative_points": ["Software updates", "Bloatware"],
                "common_issues": ["Battery degradation", "Overheating during gaming"],
                "satisfaction_score": 83
            },
            "google pixel": {
                "product": "Google Pixel",
                "positive_points": ["Camera quality", "Clean software", "Updates"],
                "negative_points": ["Battery life", "Limited availability"],
                "common_issues": ["Screen brightness", "Overheating"],
                "satisfaction_score": 81
            }
        }
        
        return feedback_db.get(product.lower(), {
            "product": product,
            "positive_points": ["Affordable", "Basic functionality"],
            "negative_points": ["Performance", "Build quality"],
            "common_issues": ["Short lifespan", "Limited support"],
            "satisfaction_score": 65
        })


# Initialize tools
product_data_tool = ProductDataTool()
market_trends_tool = MarketTrendsTool()
competitor_analysis_tool = CompetitorAnalysisTool()
customer_feedback_tool = CustomerFeedbackTool()

# Define agents
market_analyst = Agent(
    role="Market Research Analyst",
    goal="Analyze market trends and provide strategic insights",
    backstory="""You are an experienced market analyst with expertise in 
    consumer electronics. You provide detailed analysis of product performance 
    and market trends to help guide business decisions.""",
    verbose=True,
    tools=[market_trends_tool, competitor_analysis_tool]
)

product_specialist = Agent(
    role="Product Specialist",
    goal="Analyze product specifications and consumer demand",
    backstory="""You are a product specialist with deep knowledge of consumer 
    electronics. Your expertise helps companies understand product details
    and market positioning.""",
    verbose=True,
    tools=[product_data_tool, customer_feedback_tool]
)

# Define tasks
market_analysis_task = Task(
    description="""Analyze the smartphone market trends with a focus on iPhone, Samsung Galaxy, and Google Pixel.
    Be sure to include:
    1. Current trend status for all three products
    2. Popularity score interpretation
    3. Monthly search volume comparison
    4. Competitive landscape analysis
    Your analysis should be comprehensive and backed by data.
    """,
    agent=market_analyst
)

product_analysis_task = Task(
    description="""Analyze the iPhone, Samsung Galaxy, and Google Pixel product details and customer feedback.
    Focus on:
    1. Price point comparison
    2. Availability status
    3. Customer rating and satisfaction scores
    4. Key positive and negative feedback points
    Compare these products and identify their strengths and weaknesses.
    """,
    agent=product_specialist
)

# Create crew
smartphone_analysis_crew = Crew(
    agents=[market_analyst, product_specialist],
    tasks=[market_analysis_task, product_analysis_task],
    verbose=2,
    process=Process.sequential
)

# Execute crew
if __name__ == "__main__":
    result = smartphone_analysis_crew.kickoff()
    print("\n==== Custom Tools CrewAI POC Results ====\n")
    print(result) 