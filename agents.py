"""Small CrewAI agent definitions used by the MarineWise MVP.

The app uses the Groq Python client for actual model calls. These CrewAI agents
are intentionally lightweight role definitions so beginners can see how the
agent layer can be expanded later without adding a complex orchestration layer.
"""

import os

from crewai import Agent

MODEL = "groq/openai/gpt-oss-120b"


def _api_key_present() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


def make_troubleshooting_agent() -> Agent:
    """Create the manual-grounded troubleshooting specialist."""
    return Agent(
        role="Marine Engine Troubleshooting Specialist",
        goal="Explain marine engine faults using supplied manual evidence and avoid unsupported claims.",
        backstory="A careful senior marine technician who cites the source manual and page for every recommendation.",
        llm=MODEL,
        verbose=False,
        allow_delegation=False,
    )


def make_training_agent() -> Agent:
    """Create the technical training specialist."""
    return Agent(
        role="Marine Technical Training Instructor",
        goal="Turn verified marine-engine manual content into clear technician training material.",
        backstory="An instructor who explains systems step by step and distinguishes manual facts from general guidance.",
        llm=MODEL,
        verbose=False,
        allow_delegation=False,
    )


def make_assessment_agent() -> Agent:
    """Create the assessment specialist."""
    return Agent(
        role="Marine Technician Assessment Specialist",
        goal="Create and evaluate technician assessments using clear evidence and consistent scoring.",
        backstory="A training assessor who produces practical questions, answer keys, and targeted remediation.",
        llm=MODEL,
        verbose=False,
        allow_delegation=False,
    )


# Export simple role objects for future extension. They are not executed at import time.
troubleshooting_agent = make_troubleshooting_agent
training_agent = make_training_agent
assessment_agent = make_assessment_agent
