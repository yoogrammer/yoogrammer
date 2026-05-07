"""
Med-Audit Swarm
Clinical safety dashboard powered by CrewAI + Streamlit + Qwen on remote AMD MI300X (vLLM).
"""

from __future__ import annotations

from typing import Any

import streamlit as st
from crewai import Agent, Crew, Process, Task
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from a local .env file (if present).
# This keeps local development flexible and production-safe for env-based config.
load_dotenv()

# -------------------------------
# Streamlit page-level configuration
# -------------------------------
st.set_page_config(page_title="Med-Audit Swarm", layout="wide")


def build_llm() -> ChatOpenAI:
    """
    Build the LangChain ChatOpenAI client pointing to a remote vLLM OpenAI-compatible API.

    IMPORTANT:
    - model is fixed to Qwen/Qwen2.5-72B-Instruct
    - api_key is set to EMPTY for vLLM local-style auth bypass
    - base_url contains a placeholder to be replaced with your AMD instance IP
    - temperature is 0.0 for deterministic medical/safety use
    """
    return ChatOpenAI(
        model="Qwen/Qwen2.5-72B-Instruct",
        api_key="EMPTY",
        base_url="http://YOUR_AMD_INSTANCE_IP:8000/v1",
        temperature=0.0,
    )


def build_agents(llm: ChatOpenAI) -> tuple[Agent, Agent]:
    """
    Construct the two core clinical agents required by the workflow.
    """
    extractor = Agent(
        role="Clinical Data Extractor",
        goal=(
            "Extract current medications, dosages, and allergies from unstructured "
            "patient notes into a clear list."
        ),
        backstory=(
            "A meticulous pharmacist AI that never assumes dosages. "
            "If info is missing, flag it."
        ),
        llm=llm,
        verbose=False,
    )

    auditor = Agent(
        role="Drug Interaction Auditor",
        goal=(
            "Identify biochemical conflicts between existing medications and the new "
            "proposed drug."
        ),
        backstory=(
            "Expert toxicologist focused on patient safety. Look for synergistic "
            "toxicity and P450 enzyme competition. Tone is clinical and evidence-based."
        ),
        llm=llm,
        verbose=False,
    )

    return extractor, auditor


def build_tasks(
    extractor: Agent,
    auditor: Agent,
    patient_history: str,
    new_medication: str,
) -> tuple[Task, Task]:
    """
    Define the two-step sequential task flow:
    1) Extract clinical data from patient history.
    2) Audit interactions between extracted data and proposed new medication.
    """
    extraction_task = Task(
        description=(
            "Analyze the following patient history and extract:\n"
            "1) Current medications\n"
            "2) Dosages (if present)\n"
            "3) Allergies\n\n"
            "If dosage or allergy details are missing, explicitly flag them as missing.\n\n"
            f"Patient History:\n{patient_history}"
        ),
        expected_output=(
            "A structured clinical extraction with sections for medications, dosages, "
            "allergies, and missing-information flags."
        ),
        agent=extractor,
    )

    audit_task = Task(
        description=(
            "Using the extraction output from the prior task and the proposed new "
            "medication below, produce a clinical safety report.\n\n"
            f"Proposed New Medication: {new_medication}\n\n"
            "Assess likely severe drug interactions, synergistic toxicity risks, and "
            "possible CYP/P450 pathway competition. Use concise, evidence-oriented "
            "clinical language and clearly call out high-risk concerns."
        ),
        expected_output=(
            "A final safety report with interaction risks, toxicity considerations, "
            "P450 concerns, and a clear risk-focused conclusion."
        ),
        context=[extraction_task],
        agent=auditor,
    )

    return extraction_task, audit_task


def run_audit(patient_history: str, new_medication: str) -> Any:
    """
    Build and execute the sequential CrewAI workflow for the clinical audit.
    """
    llm = build_llm()
    extractor, auditor = build_agents(llm)
    extraction_task, audit_task = build_tasks(
        extractor=extractor,
        auditor=auditor,
        patient_history=patient_history,
        new_medication=new_medication,
    )

    crew = Crew(
        agents=[extractor, auditor],
        tasks=[extraction_task, audit_task],
        process=Process.sequential,
        verbose=False,
    )

    return crew.kickoff()


# -------------------------------
# UI layout
# -------------------------------
st.header("⚕️ Med-Audit Swarm: Powered by AMD MI300X")

left_col, right_col = st.columns(2)

with left_col:
    # Free-form patient chart / notes input
    patient_history_input = st.text_area(
        "Patient History",
        height=320,
        placeholder=(
            "Enter unstructured clinical notes, active medications, known allergies, "
            "and relevant history..."
        ),
    )

    # Proposed new medication to validate against current context
    new_medication_input = st.text_input(
        "Proposed New Medication",
        placeholder="e.g., Clarithromycin 500 mg BID",
    )

with right_col:
    run_clicked = st.button(
        "Run Clinical Audit",
        type="primary",
        use_container_width=True,
    )

    if run_clicked:
        # Guardrails: avoid running the workflow with empty clinical context.
        if not patient_history_input.strip() or not new_medication_input.strip():
            st.warning(
                "Please enter both Patient History and Proposed New Medication before running the audit."
            )
        else:
            with st.spinner("AMD MI300X Orchestrating Agents..."):
                result = run_audit(
                    patient_history=patient_history_input.strip(),
                    new_medication=new_medication_input.strip(),
                )

            st.subheader("🛡️ Safety Report")
            st.markdown(str(result))
