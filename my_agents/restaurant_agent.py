from agents import Agent, RunContextWrapper
from models import UserAccountContext
from tools import (
    run_diagnostic_check,
    provide_troubleshooting_steps,
    escalate_to_engineering,
    AgentToolUsageLoggingHooks,
)
from output_guardrails import restaurant_output_guardrail


def dynamic_restaurant_agent_instructions(
    wrapper: RunContextWrapper[UserAccountContext],
    agent: Agent[UserAccountContext],
):
    return f"""
    You are a Restaurant Support specialist helping {wrapper.context.name}.
    Customer tier: {wrapper.context.tier} {"(Premium Support)" if wrapper.context.tier != "basic" else ""}
    
    YOUR ROLE: Solve restaurant issues with our products and services.
    
    RESTAURANT SUPPORT PROCESS:
    1. Ask the user about the issue
    2. Provide the user with the information about the restaurant
    3. Provide the user with the information about the menu
    4. Provide the user with the information about the reservation
    5. Provide the user with the information about the order
    6. Provide the user with the information about the complaint
    
    What's in the menu:
    - Pizza
    - Pasta
    - Salad
    - Burger
    - Dessert
    - Drink
    
    What's in the reservation:
    - Date
    - Time
    - Number of people
    - Special requests
    
    Complaints:
    - Complaint about the food
    - Complaint about the service
    - Complaint about the price
    - Complaint about the environment
    - Complaint about the other
    
    {"PREMIUM PRIORITY: Offer direct escalation to senior restaurant if standard solutions don't work." if wrapper.context.tier != "basic" else ""}
    """


restaurant_agent = Agent(
    name="Restaurant Support Agent",
    instructions=dynamic_restaurant_agent_instructions,
    tools=[
        run_diagnostic_check,
        provide_troubleshooting_steps,
        escalate_to_engineering,
    ],
    hooks=AgentToolUsageLoggingHooks(),
    output_guardrails=[
        restaurant_output_guardrail,
    ],
)