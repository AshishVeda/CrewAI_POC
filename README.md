# CrewAI POC with Custom Tools

This repository contains proof-of-concept implementations of CrewAI with custom tools. The examples demonstrate how to use CrewAI to create collaborative AI agent workflows for market and product analysis.

## Overview

CrewAI provides a framework for creating autonomous AI agents that can work together to accomplish complex tasks. In these examples, we demonstrate:

1. Basic tool integration with agents
2. Advanced multi-agent collaboration
3. Custom tool classes implementation

## Project Structure

- `crew_ai_poc.py` - Basic implementation with two agents and simple tools
- `advanced_crew_poc.py` - Advanced implementation with four agents in a sequential workflow
- `custom_tools_poc.py` - Implementation using custom tool classes extending BaseTool

## Tools Implemented

The following custom tools are implemented:

1. **Fetch Product Data** - Retrieves product pricing, availability, and ratings
2. **Fetch Market Trends** - Gets trend status and popularity metrics
3. **Get Competitor Analysis** - Provides competitive analysis data
4. **Get Customer Feedback** - Returns summarized customer feedback

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

### Running the Examples

To run the basic example:

```bash
python crew_ai_poc.py
```

To run the advanced example:

```bash
python advanced_crew_poc.py
```

To run the custom tools example:

```bash
python custom_tools_poc.py
```

## Key Concepts

### Agents

Agents are specialized AI entities with specific roles, goals, and backstories. They can use tools to fulfill their tasks.

### Tasks

Tasks define what agents should accomplish. They include a description of the work to be done and which agent should perform it.

### Crew

A crew is a team of agents working on a set of tasks. The crew defines the execution process (sequential or hierarchical).

### Tools

Tools are functions that agents can use to interact with external systems or data sources. Tools can be simple functions decorated with @tool or custom classes extending BaseTool.

## Example Workflow

In these examples, the workflow typically follows this pattern:

1. Market Analyst analyzes market trends and competitive landscape
2. Product Specialist analyzes product details and customer feedback
3. (In advanced example) Marketing Strategist develops strategies based on analysis
4. (In advanced example) Business Advisor provides final recommendations

The output is a comprehensive analysis that combines insights from all agents.

## Resources

- [CrewAI Documentation](https://docs.crewai.com/)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction.html) 