from __future__ import annotations

from contingency import Contingency
from corrective_agent import AgentResult
from violation_detector import ViolationReport


def explain_result(
    contingency: Contingency,
    post_report: ViolationReport,
    agent_result: AgentResult,
    power_flow_mode: str = "dc",
) -> str:
    mode_name = "AC" if power_flow_mode == "ac" else "DC"
    failed = f"The selected N-1 event removed {contingency.label} from service."
    if post_report.is_safe:
        if power_flow_mode == "ac":
            violation = "The post-contingency case stayed inside the configured thermal and voltage limits."
        else:
            violation = "The post-contingency case stayed inside the configured DC thermal line-loading limits."
    elif not post_report.converged:
        violation = f"The post-contingency {mode_name} power flow did not converge, which is treated as unsafe."
    else:
        parts = []
        if len(post_report.overloaded_lines):
            parts.append(f"{len(post_report.overloaded_lines)} line overload(s)")
        if len(post_report.overloaded_trafos):
            parts.append(f"{len(post_report.overloaded_trafos)} transformer overload(s)")
        if len(post_report.low_voltage_buses):
            parts.append(f"{len(post_report.low_voltage_buses)} low-voltage bus violation(s)")
        if len(post_report.high_voltage_buses):
            parts.append(f"{len(post_report.high_voltage_buses)} high-voltage bus violation(s)")
        violation = "The contingency produced " + ", ".join(parts) + "."

    chosen = agent_result.chosen
    if chosen is None and post_report.is_safe:
        action = "No corrective action was needed."
        helped = "The validated post-contingency grid was already inside the configured hard limits."
    elif chosen is None:
        action = "The agent could not find a candidate action to test."
        helped = "The final grid remains unsafe because no validated corrective action was available."
    elif agent_result.path:
        path_text = "; ".join(
            f"step {step.step_number}: {step.chosen.action.description}"
            for step in agent_result.path
        )
        action = f"The agent followed this greedy path: {path_text}."
        if chosen.safe:
            helped = (
                "Pandapower validation found no remaining hard violations after the final greedy step. "
                "Each accepted step reduced the scored combination of overload and intervention cost."
            )
        else:
            helped = (
                "The path improved the scored grid state, but no fully safe endpoint was found. "
                "Further operator review or additional controls would be required."
            )
    else:
        action = f"The agent selected: {chosen.action.description}."
        if chosen.safe:
            helped = (
                "Pandapower validation found no remaining hard violations after this action. "
                "It helped by reducing the scored combination of overload and intervention cost."
            )
        else:
            helped = (
                "No fully safe candidate was found, so the agent selected the lowest-scoring converged candidate. "
                "Further operator review or additional controls would be required."
            )

    safety = (
        "Final grid status: SAFE for this offline study."
        if post_report.is_safe or (chosen and chosen.safe)
        else "Final grid status: UNSAFE or unresolved."
    )
    return "\n\n".join([failed, violation, agent_result.observation, agent_result.thought, action, helped, safety])
