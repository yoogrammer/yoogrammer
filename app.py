"""
Med-Audit Swarm
Professional-grade clinical safety dashboard powered by CrewAI + Streamlit + Qwen on AMD MI300X.
"""

from __future__ import annotations

import time
from io import BytesIO
from typing import Any, Callable

import streamlit as st
from crewai import Agent, Crew, Process, Task
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from PyPDF2 import PdfReader

# OpenAI-compatible client exceptions used by LangChain's ChatOpenAI backend.
try:
    from openai import APIConnectionError, APIError, APITimeoutError
except Exception:  # pragma: no cover - defensive fallback for older environments
    APIConnectionError = APITimeoutError = APIError = Exception

# Load environment variables from .env for local development flexibility.
load_dotenv()

# Streamlit page config required by specification.
st.set_page_config(page_title="Med-Audit Swarm", layout="wide")

MAX_VISIBLE_LOG_ENTRIES = 30
MAX_LOG_MESSAGE_LENGTH = 450
RETRY_BACKOFF_SECONDS = 2


class UILogger:
    """Simple UI logger that writes collaboration updates to sidebar + live console."""

    def __init__(self, sidebar_placeholder: st.delta_generator.DeltaGenerator, console_placeholder: st.delta_generator.DeltaGenerator) -> None:
        self.sidebar_placeholder = sidebar_placeholder
        self.console_placeholder = console_placeholder

    def _render(self) -> None:
        entries = st.session_state.get("collab_logs", [])
        if not entries:
            log_md = "_Waiting for agent collaboration..._"
        else:
            log_md = "\n\n".join(entries[-MAX_VISIBLE_LOG_ENTRIES:])  # Keep UI compact and readable.
        self.sidebar_placeholder.markdown(log_md)
        self.console_placeholder.markdown(log_md)

    def reset(self) -> None:
        st.session_state["collab_logs"] = []
        self._render()

    def log(self, speaker: str, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        cleaned = " ".join(str(message).split())
        entry = f"- `{timestamp}` **{speaker}:** {cleaned[:MAX_LOG_MESSAGE_LENGTH]}"
        st.session_state.setdefault("collab_logs", []).append(entry)
        self._render()


def build_llm() -> ChatOpenAI:
    """
    Build deterministic ChatOpenAI client against AMD-hosted vLLM OpenAI-compatible endpoint.
    """
    return ChatOpenAI(
        model="Qwen/Qwen2.5-72B-Instruct",
        # vLLM OpenAI-compatible servers often use a placeholder token for local/private access.
        api_key="EMPTY",
        # Keep placeholder exactly as requested; replace with your AMD instance IP during deployment.
        base_url="http://YOUR_AMD_INSTANCE_IP:8000/v1",
        temperature=0.0,
        timeout=60,
        max_retries=0,  # Explicit retry handling is implemented in the app workflow.
    )


def extract_text_from_pdf(uploaded_file: Any) -> str:
    """Extract text from an uploaded PDF file, preserving page flow when possible."""
    reader = PdfReader(BytesIO(uploaded_file.getvalue()))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages).strip()


def build_step_callback(logger: UILogger, speaker: str) -> Callable[..., None]:
    """Create a defensive callback for real-time thought/process traces."""

    def _callback(*args: Any, **kwargs: Any) -> None:
        if args:
            logger.log(speaker, f"Thought update: {args[0]}")
        elif kwargs:
            logger.log(speaker, f"Thought update: {kwargs}")
        else:
            logger.log(speaker, "Thought update received.")

    return _callback


def build_agents(llm: ChatOpenAI, logger: UILogger) -> tuple[Agent, Agent]:
    """Construct Extractor and Auditor agents, with Auditor equipped with web search."""
    search_tool = DuckDuckGoSearchRun()
    extractor_callback = build_step_callback(logger, "Extractor")
    auditor_callback = build_step_callback(logger, "Auditor")

    # Agent constructor compatibility can vary by CrewAI version.
    # Try using step_callback for real-time thought logging, with a safe fallback.
    try:
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
            verbose=True,
            step_callback=extractor_callback,
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
            verbose=True,
            step_callback=auditor_callback,
            tools=[search_tool],
        )
    except TypeError:
        logger.log("System", "Step callbacks unsupported in this CrewAI version; using verbose agent logs only.")
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
            verbose=True,
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
            verbose=True,
            tools=[search_tool],
        )

    return extractor, auditor


def build_tasks(
    extractor: Agent,
    auditor: Agent,
    patient_history: str,
    new_medication: str,
) -> tuple[Task, Task]:
    """
    Define sequential tasks:
    Task 1 -> Extractor consumes patient history.
    Task 2 -> Auditor consumes Task 1 output + new medication.
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
            "If uncertainty exists, use available tools to cross-reference current safety "
            "signals, then assess severe interactions, synergistic toxicity risks, and "
            "possible CYP/P450 pathway competition."
        ),
        expected_output=(
            "A final safety report with interaction risks, toxicity considerations, "
            "P450 concerns, confidence notes, and a clear risk-focused conclusion."
        ),
        context=[extraction_task],
        agent=auditor,
    )

    return extraction_task, audit_task


def run_audit_with_retry(
    patient_history: str,
    new_medication: str,
    logger: UILogger,
    max_attempts: int = 3,
) -> Any:
    """Run CrewAI flow with robust retry logic for remote vLLM connectivity issues."""
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        logger.log("System", f"Attempt {attempt}/{max_attempts}: preparing agents and tasks.")
        try:
            llm = build_llm()
            extractor, auditor = build_agents(llm=llm, logger=logger)
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
                verbose=True,
            )
            logger.log("System", "Crew kickoff started (sequential process).")
            result = crew.kickoff()
            logger.log("System", "Crew execution completed successfully.")
            return result
        except (APIConnectionError, APITimeoutError, APIError, ConnectionError, TimeoutError) as exc:
            last_error = exc
            logger.log("System", f"Endpoint issue on attempt {attempt}: {exc}")
            if attempt < max_attempts:
                logger.log("System", "Retrying after brief backoff...")
                time.sleep(RETRY_BACKOFF_SECONDS)
            else:
                break
        except Exception as exc:
            last_error = exc
            logger.log("System", f"Unhandled execution error: {exc}")
            break

    if last_error is not None:
        raise RuntimeError(
            "Unable to complete clinical audit because the AMD vLLM endpoint is unavailable or returned an error. "
            "Please verify the server is running and try again."
        ) from last_error
    raise RuntimeError("Clinical audit failed unexpectedly.")


# -------------------------------
# UI: Header + sidebar collaboration feed
# -------------------------------
st.header("⚕️ Med-Audit Swarm: Powered by AMD MI300X")
st.sidebar.header("🤝 Collaboration Feed")
sidebar_logs = st.sidebar.empty()

# Side-by-side professional layout: inputs (left) + live console/report (right).
left_col, right_col = st.columns([1, 1], gap="large")

with right_col:
    st.subheader("🧠 Live Agent Console")
    console_logs = st.empty()
    st.subheader("🛡️ Safety Report")
    report_container = st.empty()

logger = UILogger(sidebar_placeholder=sidebar_logs, console_placeholder=console_logs)
logger._render()

# Persist latest successful/attempted inputs for one-click retry UX.
st.session_state.setdefault("last_patient_history", "")
st.session_state.setdefault("last_new_medication", "")
st.session_state.setdefault("last_run_failed", False)

with left_col:
    st.subheader("📋 Clinical Inputs")
    uploaded_pdf = st.file_uploader(
        "Upload Patient History PDF (optional)",
        type=["pdf"],
        help="Upload a patient chart as PDF, or provide notes manually in the text area below.",
    )

    patient_history_input = st.text_area(
        "Patient History",
        height=260,
        placeholder=(
            "Paste unstructured clinical notes, active medications, known allergies, "
            "and relevant history..."
        ),
    )

    new_medication_input = st.text_input(
        "Proposed New Medication",
        placeholder="e.g., Clarithromycin 500 mg BID",
    )

    run_clicked = st.button(
        "Run Clinical Audit",
        type="primary",
        use_container_width=True,
    )
    retry_clicked = st.button(
        "Retry Last Audit",
        use_container_width=True,
        disabled=not st.session_state.get("last_run_failed", False),
    )


def resolve_patient_history_from_inputs(raw_text: str, pdf_file: Any, live_logger: UILogger) -> str:
    """Resolve patient history from manual input and optional PDF upload."""
    text_from_pdf = ""
    if pdf_file is not None:
        live_logger.log("System", "PDF uploaded. Extracting patient history text...")
        text_from_pdf = extract_text_from_pdf(pdf_file)
        if not text_from_pdf.strip():
            raise ValueError("Uploaded PDF could not be parsed into readable text.")
        live_logger.log("System", "PDF extraction completed.")

    # Combine sources if both are provided so clinicians can add clarifying notes.
    parts = []
    if text_from_pdf.strip():
        parts.append(f"[PDF HISTORY]\n{text_from_pdf.strip()}")
    if raw_text.strip():
        parts.append(f"[MANUAL NOTES]\n{raw_text.strip()}")
    return "\n\n".join(parts).strip()


def execute_audit(user_text: str, pdf_file: Any, medication: str) -> None:
    """Shared execution path for run/retry buttons."""
    logger.reset()
    report_container.empty()
    try:
        final_history = resolve_patient_history_from_inputs(user_text, pdf_file, logger)
    except Exception as parse_exc:
        st.session_state["last_run_failed"] = True
        st.error(f"Failed to process patient history input: {parse_exc}")
        logger.log("System", f"Input processing error: {parse_exc}")
        return

    if not final_history or not medication.strip():
        st.session_state["last_run_failed"] = True
        st.warning("Please provide patient history (text or PDF) and a proposed new medication.")
        logger.log("System", "Validation failed: missing patient history or medication input.")
        return

    st.session_state["last_patient_history"] = user_text
    st.session_state["last_new_medication"] = medication

    with st.spinner("AMD MI300X Orchestrating Agents..."):
        try:
            result = run_audit_with_retry(
                patient_history=final_history,
                new_medication=medication.strip(),
                logger=logger,
            )
            st.session_state["last_run_failed"] = False
            report_container.markdown(str(result))
        except Exception as run_exc:
            st.session_state["last_run_failed"] = True
            st.error(
                "Could not reach or complete requests against the AMD vLLM endpoint. "
                "Please verify the server and click Retry Last Audit."
            )
            logger.log("System", f"Run failed: {run_exc}")


if run_clicked:
    execute_audit(
        user_text=patient_history_input,
        pdf_file=uploaded_pdf,
        medication=new_medication_input,
    )
elif retry_clicked:
    # Retry uses the latest text+medication from previous run context.
    execute_audit(
        user_text=st.session_state.get("last_patient_history", ""),
        pdf_file=uploaded_pdf,
        medication=st.session_state.get("last_new_medication", ""),
    )
