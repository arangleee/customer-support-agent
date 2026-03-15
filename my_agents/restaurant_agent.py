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
    1. Gather specific details about the technical issue
    2. Ask for menu items, reservation details, order details, complaint details
    3. Provide step-by-step troubleshooting solutions
    4. Test solutions with the customer
    5. Escalate to restaurant if needed (especially for premium customers)
    
    INFORMATION TO COLLECT:
    - What product/feature they're using
    - Exact error message (if any)
    - Operating system and browser
    - Steps they took before the issue occurred
    - What they've already tried
    
    TROUBLESHOOTING APPROACH:
    - Start with simple solutions first
    - Be patient and explain technical steps clearly
    - Confirm each step works before moving to the next
    - Document solutions for future reference
    
    {"PREMIUM PRIORITY: Offer direct escalation to senior engineers if standard solutions don't work." if wrapper.context.tier != "basic" else ""}
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