from agents import (
    Agent,
    output_guardrail,
    Runner,
    RunContextWrapper,
    GuardrailFunctionOutput,
)
from models import RestaurantOutputGuardRailOutput, UserAccountContext


restaurant_output_guardrail_agent = Agent(
    name="Restaurant Support Guardrail",
    instructions="""
    Analyze the restaurant support response to check if it inappropriately contains:
    
    - Menu information (menu items, menu changes, menu availability)
    - Reservation information (reservation date, reservation time, reservation availability)
    - Order information (order items, order changes, order availability)
    - Complaint information (complaint items, complaint changes, complaint availability)

    """,
    output_type=RestaurantOutputGuardRailOutput,
)


@output_guardrail
async def restaurant_output_guardrail(
    wrapper: RunContextWrapper[UserAccountContext],
    agent: Agent,
    output: str,
):
    result = await Runner.run(
        restaurant_output_guardrail_agent,
        output,
        context=wrapper.context,
    )

    validation = result.final_output

    triggered = (
        validation.contains_off_topic
        or validation.contains_menu_data
        or validation.contains_reservation_data
        or validation.contains_order_data
        or validation.contains_complaint_data
        or validation.contains_account_data
    )

    return GuardrailFunctionOutput(
        output_info=validation,
        tripwire_triggered=triggered,
    )