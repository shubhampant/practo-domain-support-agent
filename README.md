# Practo Domain Support Agent (CrewAI)

**Capstone track: Practo (Healthcare).**

This repository implements a local-first support-agent prototype for answering synthetic clinic-policy questions and looking up synthetic appointment records. Part 1 — Tasks 1–5 and Part 2 — Tasks 6–10 have implementation and automated-test evidence. Task 8 is complete, including the required same-session and fresh-session transcript demonstrations. Part 3 — Tasks 11–13 have implementation and verification evidence. Part 4 — Task 14 has an implemented AutoGen review stage with deterministic mock demonstrations and integration tests. Tasks 15–16 remain pending.

> **Important simulation disclaimer:** The appointment data and all policy documents in this repository are synthetic capstone material. The fee ranges, cancellation window, turnaround times, and other policy statements are illustrative rules for this project, not official Practo policies or medical advice. Never use real patient data in this project.

## Table of contents

- [Project status](#project-status)
- [Implemented architecture](#implemented-architecture)
- [Environment and reproducibility](#environment-and-reproducibility)
- [Part 1 — Task 1: Appointment dataset](#part-1--task-1-appointment-dataset)
- [Part 1 — Task 2: Knowledge base](#part-1--task-2-knowledge-base)
- [Part 1 — Task 3: Chunking, embeddings, and ChromaDB](#part-1--task-3-chunking-embeddings-and-chromadb)
- [Part 1 — Task 4: Grounded answer generation](#part-1--task-4-grounded-answer-generation)
- [Part 1 — Task 5: Retrieval evaluation](#part-1--task-5-retrieval-evaluation)
- [Part 2 — Task 6: Appointment lookup and escalation score](#part-2--task-6-appointment-lookup-and-escalation-score)
- [Part 2 — Task 7: CrewAI multi-agent orchestration](#part-2--task-7-crewai-multi-agent-orchestration)
- [Part 2 — Task 8: Session memory](#part-2--task-8-session-memory)
- [Part 2 — Task 9: Structured response schema](#part-2--task-9-structured-response-schema)
- [Part 2 — Task 10: Guardrails](#part-2--task-10-guardrails)
- [Part 3 — Task 11: FastAPI deployment](#part-3--task-11-fastapi-deployment)
- [Part 3 — Task 12: Structured logging](#part-3--task-12-structured-logging)
- [Part 3 — Task 13: Answer-quality evaluation](#part-3--task-13-answer-quality-evaluation)
- [Part 4 — Task 14: AutoGen review stage](#part-4--task-14-autogen-review-stage)
- [Test suite](#test-suite)
- [Current evaluation summary](#current-evaluation-summary)
- [Known limitations](#known-limitations)
- [Remaining capstone work](#remaining-capstone-work)
- [Submission](#submission)

## Project status

| Capstone area | Status | Evidence |
|---|---|---|
| Part 1, Task 1 — seeded appointment dataset | Complete | `dataset.py`, `tests/test_dataset.py` |
| Part 1, Task 2 — 12-topic knowledge base | Complete | `data/knowledge_base/`, `tests/test_knowledge_base.py` |
| Part 1, Task 3 — two chunking/indexing strategies | Implemented | `chunking.py`, two ChromaDB collections |
| Part 1, Task 4 — grounded generation and fallback | Complete | `grounded_generator.py`, `retrieval.py`, `transcripts/task4_grounded_generation.txt` |
| Part 1, Task 5 — document-level precision/recall comparison | Complete | `run_evaluation.py`, `evaluation_results.md` |
| Part 2, Task 6 — appointment lookup and escalation score | Implemented and tested | `appointment_lookup.py`, `tests/test_appointment_lookup.py` |
| Part 2, Task 7 — CrewAI multi-agent orchestration | Implemented and tested | `crewai_support.py`, `tests/test_crewai_support.py`; deterministic mock LLM; three agents and `.kickoff()` |
| Part 2, Task 8 — session memory | Complete | `session_memory.py`, tests and same-session/fresh-session transcripts |
| Part 2, Task 9 — structured response schema | Implemented and tested | `response_schema.py`, schema integration tests |
| Part 2, Task 10 — guardrails | Implemented and tested | `guardrails.py`, guardrail and grounded-generation tests |
| Part 3, Task 11 — FastAPI deployment | Implemented and verified | `api.py`, `tests/test_api.py`; HTTP, WebSocket and disconnect-recovery checks |
| Part 3, Task 12 — structured logging | Implemented and tested | `api.py`, `tests/test_api.py`; JSONL events, trace IDs, timing and contact-number masking |
| Part 3, Task 13 — 15-query answer-quality evaluation | Implemented and verified | `run_answer_quality_evaluation.py`, `answer_quality_dataset.py`, `answer_quality_results.json`, `task13_report_details.txt` |
| Part 4, Task 14 — AutoGen review stage | Implemented and tested | `autogen_review.py`, `tests/test_autogen_review.py`; two-agent bounded review, Pydantic verdict, approve/revise demonstrations, CrewAI integration |
| Part 4, Task 15 — governance and budget cap | Pending | Least-autonomy enforcement, risk classification and runtime budget rejection remain to be completed |
| Part 4, Task 16 — response caching | Pending | Cache-hit demonstration and before/after evidence remain to be completed |

## Implemented architecture

```text
Synthetic appointment records
        |
        +--> dataset.py
        |      Seeded generator and validation
        |
Knowledge-base Markdown files
        |
        +--> chunking.py
               |
               +--> Fixed-size chunks (350 chars, 60-char overlap)
               |       +--> ChromaDB: knowledge_fixed_chunks
               |
               +--> Sentence-based chunks (350-char target,
                       1-sentence overlap)
                       +--> ChromaDB: knowledge_sentence_chunks
                                      |
User query --------------------------+
                                      |
                                      v
                            retrieval.py
                            top-k retrieval and
                            similarity threshold
                                      |
                                      v
                            grounded_generator.py
                            deterministic MOCK-style
                            answer generation/fallback
                                      |
                                      v
                            CrewAI support workflow
                            Retrieval Agent
                            Lookup Agent
                            Response Composer
                                      |
                                      v
                            Structured response,
                            guardrails and session memory
                                      |
                                      v
                            FastAPI endpoints
                            HTTP + WebSocket
                                      |
                                      v
                            Structured request logging
                                      |
                                      v
                            Task 13 answer-quality
                            evaluation and JSON report
```

The recommended retrieval strategy based on the current small benchmark is sentence-based chunking. It achieved higher mean document-level precision while retaining the same recall as fixed-size chunking.

## Environment and reproducibility

The development environment used for the verified runs was:

- Windows with PowerShell, including the VS Code terminal.
- Python 3.11.9 inside the project virtual environment at `.venv`.
- SentenceTransformers with a local `all-MiniLM-L6-v2` embedding model.
- ChromaDB for local vector storage.
- CrewAI for multi-agent orchestration.
- AutoGen AgentChat/Core in the isolated `.venv-autogen` environment for Task 14 review.
- FastAPI for the application API.
- Pytest for automated tests.
- Deterministic mock-model behavior for the capstone's offline-oriented demonstrations.

### 1. Open the project and activate the virtual environment

In PowerShell:

```powershell
cd D:\PractoSupportAgent
.\.venv\Scripts\Activate.ps1
python --version
```

Expected Python version:

```text
Python 3.11.9
```

If the virtual environment does not exist, create it using the Python 3.11 installation and install the dependencies from the repository's dependency file, if present:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

These are environment-setup commands. They may need network access if packages are not already cached or installed. Runtime model inference should use the locally available embedding model and should not require an API key.

For a fully offline machine, prepare dependencies and the embedding model in advance, transfer them through the approved process, and verify runtime behavior with network access disabled.

### 2. Run the automated tests

```powershell
python -m pytest -v
```

The latest full-suite run after Task 14 integration reported **115 passed, 0 failed**, with 661 CrewAI deprecation warnings. Re-run the suite after further code changes to establish the current result.

Warnings are not test failures, but they should be reviewed when upgrading CrewAI or related dependencies.

### 3. Reproduce the dataset generation

```powershell
python dataset.py
```

The generator uses a fixed seed, so the records and reported statistics are deterministic for the same code and dependencies.

### 4. Reproduce the grounded-generation demonstration

```powershell
python grounded_generator.py
```

This demonstrates five in-scope policy questions and an out-of-scope FIFA World Cup question. The out-of-scope question should return an insufficient-evidence fallback rather than inventing an answer.

The saved demonstration transcript is:

```text
transcripts/task4_grounded_generation.txt
```

### 5. Reproduce the retrieval evaluation

```powershell
python run_evaluation.py
```

This evaluates the same five in-scope questions against both chunking collections, prints per-query document-level precision and recall, reports mean metrics, and prints top-1 cosine similarities for five in-scope and two out-of-scope queries.

The detailed written report is:

```text
evaluation_results.md
```

### Rebuilding the vector index

The two local ChromaDB collections are stored under `data/chroma_db`:

- `knowledge_fixed_chunks`
- `knowledge_sentence_chunks`

The verified index-building entry point is `vector_store.py`. Run these commands from the project root (`D:\PractoSupportAgent`) in the activated virtual environment.

First, check whether the local model folder exists:

```powershell
Test-Path .\models\all-MiniLM-L6-v2
```

If this returns `False`, download and save the embedding model while network access is available, then transfer the saved model through the approved process if the target environment is offline:

```powershell
python download_model.py
```

**Do not run `download_model.py` as an offline recovery step if the model is missing:** its initial `SentenceTransformer(MODEL_NAME)` call may attempt to access Hugging Face.

Rebuild both collections:

```powershell
python vector_store.py
```

This command loads the knowledge documents, requires at least 12 documents, loads the embedding model with `local_files_only=True`, regenerates both chunk strategies and embeddings, and prints each collection's indexed chunk count. It also runs a sample appointment-cancellation search and prints the top results.

**Destructive rebuild behavior:** `rebuild_collection()` deletes each named collection before recreating it. Running `python vector_store.py` therefore replaces the contents of both `knowledge_fixed_chunks` and `knowledge_sentence_chunks` in `data/chroma_db`. Back up the database first if it contains any data that must be retained. The builder uses `collection.upsert()` to write the new chunk records.

## Part 1 — Task 1: Appointment dataset

### Design choices

`dataset.py` generates **50 synthetic appointment records** using `SEED = 42`.

- **Categories:** `General Medicine`, `Cardiology`, `Dermatology`, `Pediatrics`, and `Orthopedics`.
- **Statuses:** `Scheduled`, `Completed`, `Cancelled`, `No-Show`, and `Rescheduled`.
- The category and status lists are balanced before deterministic shuffling, giving 10 records per category and 10 per status.
- `days_since_created` is an integer from 0 through 30.
- `follow_up_required` is generated with a probability of 0.20; the generated dataset contains **11/50 records (22%)**, within the required 10–30% range.
- Record IDs are unique and the generator is deterministic for the fixed seed.

### Synthetic consultation-fee ranges

| Specialty | Fee range (INR) |
|---|---:|
| General Medicine | ₹300–₹800 |
| Cardiology | ₹600–₹1,800 |
| Dermatology | ₹400–₹1,200 |
| Pediatrics | ₹300–₹900 |
| Orthopedics | ₹500–₹1,500 |

These ranges are illustrative values selected for the capstone to give different specialties plausible relative fee bands; they are not official Practo prices.

### Reproduce and validate

```powershell
python dataset.py
python -m pytest tests/test_dataset.py -v
```

The dataset tests validate record count, deterministic generation, unique IDs, category/status coverage, follow-up percentage, fee ranges, and field constraints.

## Part 1 — Task 2: Knowledge base

The knowledge base is in `data/knowledge_base/`. It contains 12 Markdown documents, each written as a short synthetic policy description with 2–5 sentences.

| File | Topic |
|---|---|
| `appointment_booking.md` | Appointment booking |
| `cancellation_rescheduling.md` | Cancellation and rescheduling window |
| `consultation_fees.md` | Consultation fees by specialty |
| `insurance_claims.md` | Insurance-claim process |
| `prescription_refills.md` | Prescription-refill policy |
| `lab_test_turnaround.md` | Lab-test turnaround times |
| `telemedicine.md` | Telemedicine eligibility |
| `emergency_protocol.md` | Emergency-visit protocol |
| `patient_privacy.md` | Patient-data privacy |
| `follow_up_discount.md` | Follow-up-visit discount |
| `second_opinion.md` | Second-opinion process |
| `home_visits.md` | Home-visit eligibility |

All policy values are capstone simulation assumptions. They must not be represented as official Practo policies.

### Reproduce and validate

```powershell
python -m pytest tests/test_knowledge_base.py -v
```

The tests verify required documents, content presence, sentence-count constraints, synthetic-policy labeling, and the stated scope of the privacy masker.

## Part 1 — Task 3: Chunking, embeddings, and ChromaDB

`chunking.py` loads the knowledge-base documents, creates chunk records, and preserves source metadata so retrieved chunks can be mapped back to their parent Markdown files.

Two strategies are implemented:

1. **Fixed-size with overlap:** 350-character chunks with a 60-character overlap.
2. **Sentence-based:** chunks target up to 350 characters and overlap by one sentence.

Both strategies use the local SentenceTransformers model `all-MiniLM-L6-v2` (384-dimensional embeddings) and separate ChromaDB collections:

| Strategy | Collection | Current chunk count |
|---|---|---:|
| Fixed-size | `knowledge_fixed_chunks` | 26 |
| Sentence-based | `knowledge_sentence_chunks` | 36 |

The index builder uses `collection.upsert()` so chunks can be inserted or updated by stable IDs. ChromaDB data is stored locally in `data/chroma_db`.

### Reproduce and validate

```powershell
python -m pytest tests/test_chunking.py -v
```

The tests check that knowledge documents load, fixed-size chunk limits and overlap are respected, sentence splitting preserves sentences, source metadata is present, and invalid chunk settings are rejected.

> **Offline note:** The embedding model must already exist at `models/all-MiniLM-L6-v2` for offline operation. A first-time download or package installation may require connectivity; this is a setup step, not an intended runtime dependency.

## Part 1 — Task 4: Grounded answer generation

`retrieval.py` retrieves relevant chunks and applies a similarity threshold. `grounded_generator.py` combines retrieved evidence into a deterministic answer and returns an insufficient-evidence fallback when the top-1 similarity is below the threshold. Answers are restricted to information supported by the retrieved context.

### Empirical threshold calibration

The configured threshold in `retrieval.py` is:

```python
SIMILARITY_THRESHOLD = 0.4026
```

The threshold was derived from observed cosine similarities, rather than selecting a generic preset.

| In-scope query | Top source | Cosine similarity |
|---|---|---:|
| Appointment cancellation window | `cancellation_rescheduling.md` | 0.6715 |
| Cardiology fee range | `consultation_fees.md` | 0.7338 |
| Routine blood-test turnaround | `lab_test_turnaround.md` | 0.6051 |
| Telemedicine eligibility | `telemedicine.md` | 0.5815 |
| Second-opinion process | `second_opinion.md` | 0.8110 |

| Out-of-scope query | Top source returned by retrieval | Cosine similarity |
|---|---|---:|
| FIFA World Cup 2018 winner | `lab_test_turnaround.md` | 0.1118 |
| Current Bitcoin price | `consultation_fees.md` | 0.2236 |

The lowest observed in-scope similarity is **0.5815** and the highest observed out-of-scope similarity is **0.2236**.

```text
Candidate threshold = (0.5815 + 0.2236) / 2
                    = 0.40255
Rounded threshold   = 0.4026
```

The configured threshold is therefore between the two observed groups in this small sample. This is a provisional calibration based on seven queries, not a guarantee of correct classification for all future questions.

### Demonstration and tests

```powershell
python grounded_generator.py
python -m pytest tests/test_retrieval.py tests/test_grounded_generator.py -v
```

The transcript at `transcripts/task4_grounded_generation.txt` records the five in-scope demonstrations and the FIFA out-of-scope fallback. The fallback response is:

> I don't know. The available knowledge base does not contain sufficient supporting information.

## Part 1 — Task 5: Retrieval evaluation

`run_evaluation.py` evaluates the same five in-scope queries against both ChromaDB collections. Retrieved chunks are mapped to parent document filenames and duplicate filenames are removed before document-level scoring.

- **Precision** = relevant retrieved parent documents / all retrieved parent documents.
- **Recall** = relevant retrieved parent documents / all relevant parent documents.

Full per-query retrieved-document lists and arithmetic are recorded in `evaluation_results.md`.

### Results

| Metric | Fixed-size chunks | Sentence-based chunks |
|---|---:|---:|
| Mean precision | 0.500 | 0.900 |
| Mean recall | 1.000 | 1.000 |
| Queries retrieving the relevant document | 5/5 | 5/5 |

**Recommendation:** Use sentence-based chunking for the current capstone because its mean precision is **0.900**, versus **0.500** for fixed-size chunking, while both achieve **1.000 mean recall**. That is an absolute precision gain of **0.400 (40 percentage points)** across the five tested queries. The benchmark is small, and the sentence-based telemedicine query still retrieves `emergency_protocol.md` as an additional non-relevant document.

### Reproduce the evaluation

```powershell
python run_evaluation.py
```

Review `evaluation_results.md` for both strategies' per-query arithmetic, the numeric recommendation, threshold calibration, and limitations.

## Part 2 — Task 6: Appointment lookup and escalation score

### Implementation

`appointment_lookup.py` implements:

- `check_appointment_status(record_id: str)`: searches the deterministic synthetic `APPOINTMENTS` records from `dataset.py` and returns whether the record was found. For a match it returns the record ID, status, consultation fee in INR, escalation score, and whether escalation is recommended.
- `calculate_escalation_score(follow_up_required, days_since_created)`: validates its inputs and computes a deterministic score between 0 and 1.

The tool normalizes IDs by trimming surrounding whitespace and converting to uppercase. An unknown ID returns a structured not-found result instead of invented appointment data. Invalid empty IDs, invalid flag types, and recency values outside 0–30 are rejected.

### Score formula and threshold

The score uses two equally weighted signals:

```text
follow_up_signal = 1 if follow_up_required else 0
recency_signal = days_since_created / 30
score = 0.5 * follow_up_signal + 0.5 * recency_signal
```

The escalation threshold is a **design-selected synthetic operational rule** of `0.8`, not a clinically validated threshold and not a measured 80th-percentile threshold. Since records without a follow-up flag can score at most 0.5, a record can meet the threshold only when follow-up is required and `days_since_created >= 18`. This rule is for capstone demonstration only; it does not represent medical urgency or official Practo policy.

### Reproduce and verify

Run the Task 6 tests:

```powershell
python -m pytest tests/test_appointment_lookup.py -v
```

Expected verified result: `11 passed`.

Run the complete regression suite:

```powershell
python -m pytest -v
```

The latest previously reported complete-suite result after Tasks 11–12 was **110 passed, 0 failed**, with 661 CrewAI deprecation warnings.

To reproduce the escalation count for the current deterministic dataset:

```powershell
python -c "from dataset import APPOINTMENTS; from appointment_lookup import check_appointment_status; results = [check_appointment_status(a['record_id']) for a in APPOINTMENTS]; escalated = [r for r in results if r['escalation_recommended']]; print('Total appointments:', len(results)); print('Escalation threshold: 0.8'); print('Escalated:', len(escalated)); print('Not escalated:', len(results) - len(escalated)); [print(r['record_id'], '| score:', r['escalation_score'], '| status:', r['status']) for r in escalated]"
```

The previously verified result for the seed-42 dataset was **7 of 50 records (14%)** flagged for operational escalation and 43 not flagged. The seven IDs were `APT-0010`, `APT-0011`, `APT-0020`, `APT-0022`, `APT-0023`, `APT-0024`, and `APT-0050`. These counts depend on the dataset and scoring rule remaining unchanged.

The corresponding tests are in `tests/test_appointment_lookup.py` and cover score boundaries, score range, lookup success, case/whitespace normalization, unknown IDs, and invalid inputs.

## Part 2 — Task 7: CrewAI multi-agent orchestration

`crewai_support.py` builds a sequential CrewAI crew with three agents: Retrieval Agent, Lookup Agent, and Response Composer. The Retrieval Agent has access to the RAG lookup tool; the Lookup Agent has access to the appointment-status lookup tool. The crew runs through `.kickoff()` and uses the project's deterministic `DeterministicMockLLM` rather than a hosted inference API.

Tool invocations and tool results are captured for test/audit inspection. The crew is configured with `tracing=False` and `memory=False`; session history is handled separately by the project's in-process memory implementation.

**Do not treat these settings alone as proof of zero network traffic.** CrewAI telemetry must be disabled before kickoff and runtime network behavior independently verified for a strict offline claim.

Verify:

```powershell
python -m pytest tests/test_crewai_support.py -v
```

## Part 2 — Task 8: Session memory

`session_memory.py` provides thread-safe, in-process session history with isolated session IDs, bounded turns, history retrieval, and clear-session operations. `crewai_support.py` uses this history to resolve supported follow-up questions, including cancellation-related follow-ups. Tests cover memory storage and CrewAI integration, including session isolation and one-shot sessions when no session ID is supplied.

Verify:

```powershell
python -m pytest tests/test_session_memory.py tests/test_crewai_session_memory.py -v
```

This memory is intentionally process-local and does not survive a restart. The implementation and tests are complete, and both objective-required demonstrations have been generated and inspected:

- `transcripts/task8_same_session.txt` demonstrates two turns using the same session ID. The second turn receives the previous cancellation-policy conversation as history and resolves the follow-up in that context.
- `transcripts/task8_fresh_session.txt` demonstrates the follow-up-style question in a separate session whose initial history is empty. It answers from the knowledge base without relying on the earlier conversation.

Both transcripts record the session IDs, history before/after the turns, tool invocation/results, and verification outcome. The same-session transcript verifies that both turns use `task8-demo-same-session`; the fresh-session transcript verifies that `task8-demo-fresh-session` starts with no prior history. **Task 8 acceptance transcript evidence is complete.**

Recreate the transcripts from PowerShell:

```powershell
python demo_task8_memory.py
Get-Content .\transcripts\task8_same_session.txt
Get-Content .\transcripts\task8_fresh_session.txt
```

## Part 2 — Task 9: Structured response schema

`response_schema.py` defines a Pydantic `SupportResponse` model with constrained `mode` and `status` fields, answer text, sources, and optional appointment lookup fields. Extra fields are forbidden; blank answers and invalid fee/score values are rejected. Successful lookup responses require a record ID. `crewai_support.py` returns the validated model as a dictionary in the `structured_response` field while retaining the existing output and tool-audit fields.

Verify:

```powershell
python -m pytest tests/test_response_schema.py tests/test_crewai_structured_response.py -v
```

## Part 2 — Task 10: Guardrails

`guardrails.py` implements the following safeguards:

- **Fixed-format contact-number masking:** Indian mobile numbers, optionally prefixed with `+91`, are replaced with `[MASKED_CONTACT_NUMBER]` before RAG processing and session storage. This masker is not intended to identify arbitrary names, diagnoses, or insurance IDs in free text.
- **Prompt-injection detection:** suspicious instruction-override requests are rejected before CrewAI execution; the rejected raw query is not added to session history.
- **Output groundedness check:** generated answer sentences are checked against retrieved evidence. Unsupported claims cause `GroundedSupportAgent.ask()` to return `insufficient_evidence` and a refusal message.

The groundedness check is a deterministic lexical-overlap heuristic, not a semantic fact-checker and not a guarantee against all hallucinations. Use only fabricated test examples; never use real patient information.

Verify the deliberate guardrail test cases and integration:

```powershell
python -m pytest tests/test_guardrails.py tests/test_crewai_guardrails.py tests/test_grounded_generator.py -v
```

The tests demonstrate contact masking, injection rejection before crew execution, and rejection of a deliberately unsupported generated answer.

## Part 3 — Task 11: FastAPI deployment

`api.py` exposes the support-agent application through FastAPI.

### Start the server

Activate the project virtual environment, then run this command from the project root:

```powershell
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

The server listens on `http://127.0.0.1:8000` on the local machine. Keep the server terminal open while testing. To stop it, press `Ctrl+C`.

### HTTP endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/ask` | Submit a support query to the RAG support crew |
| `POST` | `/appointment-status` | Look up a synthetic appointment record |

Example health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Example support request:

```powershell
$body = @{ query = "What is the cancellation policy?" } |
    ConvertTo-Json

Invoke-RestMethod `
    -Uri http://127.0.0.1:8000/ask `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

Example appointment lookup:

```powershell
$body = @{ record_id = "APT-0010" } | ConvertTo-Json

Invoke-RestMethod `
    -Uri http://127.0.0.1:8000/appointment-status `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

### WebSocket endpoint

`ws://127.0.0.1:8000/ws/chat` supports multiple JSON messages over the same connection. Send a message such as:

```json
{"query": "What is the cancellation policy?"}
```

A successful response uses the `message` event type and includes a request ID, session ID, and structured support response. The server manages the WebSocket session ID. Invalid messages return an error event; a disconnected client should not prevent other clients from using the service.

### Task 11 verification evidence

- `python -m pytest tests/test_api.py -v`: 11 tests passed in the previously reported verification.
- Local checks passed for `/health`, `/appointment-status`, and `/ask`.
- A live WebSocket test passed for two turns on one connection with a stable session ID.
- A live disconnect-recovery test passed: the health endpoint remained available and a new WebSocket client received a response.

These checks verify the tested Task 11 behavior; they are not a production load or security assessment.

## Part 3 — Task 12: Structured logging

Task 12 adds structured request logging to the FastAPI application in `api.py`.

The implementation records structured JSON Lines (JSONL) log events and includes request trace identifiers and timing information. The tested logging behavior also includes HTTP request logging and fixed-format contact-number masking.

### Logging capabilities

- Structured JSONL events for covered requests.
- Trace IDs for correlating request-related events.
- Timing information for request processing.
- HTTP request logging.
- Fixed-format contact-number masking for covered patterns.
- Automated tests for the implemented logging and masking behavior.

### Verify Task 12

Run:

```powershell
python -m pytest tests/test_api.py -v
```

The previously reported Task 12/full-suite verification showed the API tests passing and a full-suite result of **110 passed, 0 failed**, with 661 CrewAI deprecation warnings.

### Task 12 limitations

The contact-number masker is fixed-format and does not identify every possible type of sensitive information. The implementation does not establish production-ready log retention, rotation, access controls, or a dedicated production logging destination. Logging tests cover the implemented behavior and should not be treated as a complete privacy or compliance audit.

## Part 3 — Task 13: Answer-quality evaluation

Task 13 implements a 15-query answer-quality evaluation under `MOCK_LLM`, reporting **Accuracy, Grounding, Completeness, and Safety** for every query and calculating the average of each metric across all 15 queries.

### Files

| File | Purpose |
|---|---|
| `answer_quality_dataset.py` | Defines the 15 evaluation cases, expected facts, and safety requirements |
| `run_answer_quality_evaluation.py` | Runs the support crew, scores its responses, calculates averages, and writes the report |
| `answer_quality_results.json` | Machine-readable evaluation report with per-query scores and metadata |
| `task13_report_details.txt` | Detailed UTF-8 text/JSON export for inspection |
| `patch_task13_judge.py` | Development patch script used during the evaluator work |
| `patch_task13_safety.py` | Development patch script used during safety-pattern refinement |
| `patch_task13_insurance.py` | Development patch script used during insurance-fact refinement |

The patch scripts and backup files are development artifacts; the main evaluation entry point is `run_answer_quality_evaluation.py`.

### Evaluation dataset coverage

The evaluation dataset contains 15 cases:

| Case | Topic | Coverage |
|---:|---|---|
| 1 | Appointment booking | Required KB topic |
| 2 | Cancellation and rescheduling | Required KB topic |
| 3 | Consultation fees | Required KB topic |
| 4 | Emergency protocol | Required KB topic |
| 5 | Follow-up discount | Required KB topic |
| 6 | Home visits | Required KB topic |
| 7 | Insurance claims | Required KB topic |
| 8 | Lab-test turnaround | Required KB topic |
| 9 | Patient privacy | Required KB topic |
| 10 | Prescription refills | Required KB topic |
| 11 | Second opinion | Required KB topic |
| 12 | Telemedicine | Required KB topic |
| 13 | Cancellation edge case | Edge case: completed appointments cannot be rescheduled |
| 14 | Out-of-scope sports | Out of scope: FIFA World Cup 2018 winner |
| 15 | Out-of-scope finance | Out of scope: current Bitcoin price |

This covers all 12 required knowledge-base topics, one deliberately designed edge case, and two out-of-scope questions.

The out-of-scope cases test whether the support system recognizes insufficient evidence instead of fabricating an answer from an unrelated healthcare knowledge base.

### Scoring rubric

Each property is scored on a 0–2 scale.

| Score | Interpretation |
|---:|---|
| 0 | Fail |
| 1 | Partial |
| 2 | Pass |

The four properties are defined as follows:

- **Accuracy:** Checks expected facts and the retrieved result for supported factual content and material contradictions.
- **Grounding:** Checks retrieval status and source metadata, including similarity and threshold information.
- **Completeness:** Checks how many expected facts needed to answer the query are present in the response.
- **Safety:** Checks for unsafe guarantees, specified safety signals, and unsafe handling patterns covered by the evaluator.

Each query's report includes its scores and score rationales. The report also records the retrieval metadata and expected-fact checks used by the deterministic evaluator.

### Run Task 13

From the project root, with the virtual environment activated:

```powershell
python .\run_answer_quality_evaluation.py
```

The script runs the 15 test queries through the support crew in `rag` mode, uses a separate session ID for each evaluation case, evaluates the resulting answers, calculates metric averages, and writes:

```text
answer_quality_results.json
```

The detailed export used for inspection is:

```text
task13_report_details.txt
```

### Latest verified results

The latest reported evaluation produced the following averages:

| Metric | Average score | Maximum |
|---|---:|---:|
| Accuracy | 1.800 | 2 |
| Grounding | 2.000 | 2 |
| Completeness | 1.800 | 2 |
| Safety | 1.867 | 2 |

The reported overall normalized score is **93.34%**. Human review is required.

These are the results recorded by the current deterministic scoring implementation. They are not an independently validated semantic-quality benchmark.

### Validate the report structure

After running the evaluation, use the following PowerShell command to validate the report's query count, score keys, rationale keys, averages, and judge metadata:

```powershell
@'
import json
from pathlib import Path

path = Path("answer_quality_results.json")
report = json.loads(path.read_text(encoding="utf-8"))
results = report.get("results", [])

required = {"accuracy", "grounding", "completeness", "safety"}

print("Query count:", len(results))
print("Reported query count:", report.get("query_count"))
print("Metric averages:", report.get("metric_averages"))

print(
    "All 15 records have four scores:",
    len(results) == 15 and all(
        set(item.get("scores", {})) == required
        for item in results
    )
)

print(
    "All records have four rationales:",
    len(results) == 15 and all(
        set(item.get("score_rationales", {})) == required
        for item in results
    )
)

print("Judge mode:", report.get("judge_mode"))
print("Judge type:", report.get("judge_type"))
'@ | python -
```

The previously verified output was:

```text
Query count: 15
Reported query count: 15
Metric averages: {'accuracy': 1.8, 'grounding': 2, 'completeness': 1.8, 'safety': 1.867}
All 15 records have four scores: True
All records have four rationales: True
Judge mode: MOCK_LLM
Judge type: task13_deterministic_mock_llm
```

Run the evaluation again before submission if any code, dataset, or scoring rules change.

### Important evaluator limitation

The current Task 13 judge is **deterministic and pattern-based, not a genuine semantic LLM judge**.

The evaluator supplies a rubric prompt and uses explicit, auditable rules and pattern matching to simulate the four scoring dimensions. The metadata identifies the evaluation mode as `MOCK_LLM` and the judge type as `task13_deterministic_mock_llm`.

The separate CrewAI support crew also uses a deterministic mock model. That mock model is used for the support workflow; it should not be confused with a model that semantically evaluates answer quality.

The scores have the following limitations:

1. Regex matches do not prove that a statement is semantically entailed by its source.
2. A response may express a correct fact using wording that the patterns do not recognize.
3. Grounding is approximated from retrieval status, source filenames, similarity and the configured threshold. It does not independently verify every answer claim against the raw evidence text.
4. Safety scoring detects only the unsafe patterns and safety signals explicitly encoded in the evaluator. It is not a comprehensive clinical safety assessment.
5. Out-of-scope scoring checks the evaluator's defined insufficient-evidence behavior; a high score does not establish general out-of-domain reliability.

The report records these limitations and sets `human_review_required` to `true`. **Human review is required before treating the scores as evidence of production answer quality or healthcare safety.**

The evaluator should be described accurately in demonstrations and submission material: it is a deterministic mock evaluation harness operating in `MOCK_LLM` mode, not a semantic LLM-as-judge implementation.

## Part 4 — Task 14: AutoGen review stage

Task 14 adds a bounded AutoGen review step after the existing CrewAI support workflow drafts an answer. The existing three-agent CrewAI workflow remains in place. For an answerable RAG response, `crewai_support.py` passes the user's query, the CrewAI draft answer, and the retrieved evidence text to `autogen_review.py`.

### Design

- `autogen_review.py` defines the Pydantic `ReviewVerdict` schema:
  - `approved: bool`
  - `final_answer: str`
  - `reason: str`
- The AutoGen team uses `RoundRobinGroupChat` with `max_turns=2`.
- The two agents are `Policy_Compliance_Reviewer` and `Final_Editor`.
- The Final-Editor uses `output_content_type=ReviewVerdict`.
- The team registers `StructuredMessage[ReviewVerdict]` in `custom_message_types`.
- The review input includes the original query, the draft, and the retrieved context.
- The deterministic mock implementation allows the demonstrations and tests to run without a hosted LLM API key. AutoGen dependencies are isolated in `.venv-autogen` so they do not disturb the existing `.venv` used by CrewAI.

When the verdict is returned, `crewai_support.py` updates the structured answer and output with `final_answer` and exposes review metadata, including the verdict, turn count, and message trace. If the review subprocess fails, the integration preserves the original grounded answer and records the review failure instead of silently replacing the answer.

### Run the two deterministic demonstrations

From the project root in PowerShell:

```powershell
$env:OPENAI_API_KEY = ""
$env:CREWAI_DISABLE_TELEMETRY = "true"
.\.venv-autogen\Scripts\python.exe .\autogen_review.py
```

The demonstration covers two cases:

1. **Approve unchanged:** a draft that states the synthetic cancellation/rescheduling window is at least 24 hours before the appointment. The verdict approves the draft and retains the same answer.
2. **Revise unsupported claim:** a draft includes “Our clinic guarantees a full refund for every cancellation,” but the supplied context does not support that guarantee. The verdict rejects the draft as written and removes the unsupported guarantee from the final answer.

The script reports the structured verdict, the two-turn message trace, and the turn count. These are deterministic mock demonstrations, not proof of general-purpose semantic reasoning.

### Run Task 14 tests

```powershell
.\.venv-autogen\Scripts\python.exe -m pytest .\tests\test_autogen_review.py -v
```

Previously verified result:

```text
4 passed
```

The tests cover approval without changing a supported draft, removal of the configured unsupported refund-guarantee phrase, the reviewer/editor message trace, and execution without `OPENAI_API_KEY`.

Run the integration/regression checks from the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest .\tests\test_grounded_generator.py .\tests\test_retrieval.py .\tests\test_autogen_review.py -q
.\.venv\Scripts\python.exe -m pytest .\tests\test_appointment_lookup.py -v
.\.venv\Scripts\python.exe -m pytest .\tests -q
```

Previously reported results after Task 14 integration:

- Grounded-generation, retrieval, and AutoGen review tests: **18 passed**.
- Appointment lookup tests: **11 passed**.
- Full test suite after Task 14 integration: **115 passed, 0 failed**, with 661 CrewAI deprecation warnings.

These are the previously observed results; rerun the commands after any further changes to establish the current state.

### Dependency isolation

AutoGen is installed in `.venv-autogen` with `autogen-agentchat==0.7.5` and `autogen-core==0.7.5`. The original `.venv` is preserved for the CrewAI application. `.venv-autogen/` is ignored by Git.

The CrewAI integration launches the AutoGen review in a separate Python process using `.venv-autogen\Scripts\python.exe` and JSON over standard input/output. This avoids the known dependency conflict between the AutoGen protobuf requirement and the protobuf version in the original environment.

### Limitations

- The deterministic mock reviewer recognizes a limited set of configured refund-guarantee phrases. It is **not** a general semantic policy-compliance engine and may miss unsupported claims expressed in other wording.
- The supported-draft approval case verifies the configured demonstration, not every possible answer or policy.
- The review process is bounded to two AutoGen turns; that is a workflow bound, not a guarantee of factual correctness.
- A mock LLM and disabled CrewAI telemetry do not, by themselves, prove zero network activity. Strict offline acceptance still requires testing with network access disabled or independently monitoring network activity.
- The automated test file is the current reproducible evidence. Do not claim that separate saved Task 14 transcript files exist unless they have actually been generated and checked into the repository.

### Integrated CrewAI + AutoGen Demonstration

Task 14 includes a reproducible integration demonstration that runs the existing CrewAI support workflow and verifies the subsequent AutoGen policy-review stage.

**Workflow**

1. The CrewAI Retrieval Agent retrieves relevant knowledge-base context.
2. The Response Composer produces the support response.
3. The AutoGen `Policy_Compliance_Reviewer` evaluates the draft against the original retrieved context.
4. The AutoGen `Final_Editor` produces a structured `ReviewVerdict`.
5. The reviewed answer is returned as the final response.

**Demonstration script:** `scripts/demo_task14_integration.py`

Run these commands from the project root in PowerShell:

```powershell
$env:OPENAI_API_KEY = ""
$env:CREWAI_DISABLE_TELEMETRY = "true"
$env:CREWAI_TRACING_ENABLED = "false"
$env:OTEL_SDK_DISABLED = "true"

.\.venv\Scripts\python.exe .\scripts\demo_task14_integration.py
```

The script checks that:

- The CrewAI RAG workflow executes.
- The AutoGen review completes.
- The review verdict contains a Boolean approval decision.
- The review is bounded to two turns.
- The reviewed answer matches the returned final output.
- The structured response matches the final output.
- Both AutoGen agents appear in the review trace.

Successful execution ends with:

```text
DEMONSTRATION PASSED: integrated workflow checks succeeded.
```

**Run the handoff integration test:**

```powershell
.\.venv\Scripts\python.exe -m pytest .\tests\test_crewai_support.py::test_task14_passes_composer_output_and_original_context_to_autogen -v
```

This test verifies that the actual CrewAI kickoff output is supplied as the AutoGen draft, the original retrieved context is preserved, and the reviewed answer is reflected in the final response.

**Run the AutoGen review tests:**

```powershell
.\.venv-autogen\Scripts\python.exe -m pytest .\tests\test_autogen_review.py -v
```

**Execution environment:** CrewAI runs in `.venv`; AutoGen runs in the isolated `.venv-autogen` environment. The demonstration uses the configured deterministic mock components and does not require a live API key for the intended mock workflow.

**Important limitation:** The deterministic AutoGen mock detects a limited set of configured unsupported refund-guarantee phrases. Passing these tests demonstrates the implemented workflow and configured test cases, not comprehensive semantic policy compliance. Setting an empty API key does not, by itself, prove that all dependencies or execution paths work without network access.


## Test suite

Run the complete test suite from the repository root:

```powershell
python -m pytest -v
```

The latest verified full-suite run after Task 14 integration was:

```text
115 passed, 0 failed
661 CrewAI deprecation warnings
```

This result was verified after adding the Composer-to-AutoGen handoff integration test and the integrated demonstration script.

This is the latest recorded full-suite result after Task 14 integration; rerun the tests to establish the current result after any further changes.

Dependency consistency check:

```powershell
python -m pip check
```

The previously reported result was:

```text
No broken requirements found.
```

The tests cover dataset generation, chunking, knowledge-base structure, retrieval behavior, grounded generation, appointment lookup, CrewAI orchestration, session memory, structured responses, guardrails, and the FastAPI application.

Passing tests do not by themselves prove every objective requirement or replace the required demonstration transcripts and evaluation reports.

## Current evaluation summary

- **Dataset:** 50 synthetic appointment records, five categories, five statuses, and 22% follow-up-required records.
- **Knowledge base:** 12 required topic documents.
- **Chunking:** 26 fixed-size chunks and 36 sentence-based chunks in separate collections.
- **Embeddings:** Local `all-MiniLM-L6-v2`, 384 dimensions.
- **Retrieval threshold:** 0.4026, empirically selected from five in-scope and two out-of-scope query measurements.
- **Grounded generation:** Five in-scope demonstrations and an out-of-scope fallback.
- **Retrieval benchmark:** Fixed-size precision/recall = 0.500/1.000; sentence-based = 0.900/1.000.
- **Task 6 escalation rule:** Threshold 0.8; the previously verified seed-42 dataset flagged 7/50 synthetic appointments (14%).
- **Task 8 session memory:** In-process memory with same-session and fresh-session transcript demonstrations.
- **Task 11 API:** HTTP endpoints and WebSocket behavior tested, including disconnect recovery.
- **Task 12 logging:** Structured JSONL events, trace IDs, timing and contact-number masking implemented and tested.
- **Task 13 evaluation:** 15 queries, four scores and four score rationales per query, with averages reported.
- **Task 13 averages:** Accuracy 1.800/2; Grounding 2.000/2; Completeness 1.800/2; Safety 1.867/2.
- **Task 13 overall normalized score:** 93.34% in the latest reported run.
- **Task 13 review status:** Human review required because scoring is deterministic and pattern-based.
- **Task 14 AutoGen review:** Two-agent, two-turn review stage with structured Pydantic verdicts, deterministic approval/revision examples, and CrewAI RAG integration. The mock reviewer only detects configured phrases.
- **Task 14 tests:** 4 focused AutoGen review tests passed, the Composer-to-AutoGen handoff integration test passed, and the latest full suite reported **115 passed**.
- **Part 4 remaining:** Task 15 governance/budget-cap controls and Task 16 response caching remain pending.

## Known limitations

1. The policy documents and appointment dataset are synthetic and do not represent official Practo data or policies.
2. Similarity-threshold calibration uses only seven measured queries; additional varied in-scope and out-of-scope examples are needed.
3. The retrieval evaluation contains five in-scope questions, so its precision/recall estimates should not be treated as broad performance guarantees.
4. Retrieval metrics do not measure the correctness, completeness, or safety of every generated answer.
5. The embedding model and dependencies must be prepared locally before running in a completely offline environment.
6. Task 6 escalation is a synthetic operational rule, not a clinically validated measure.
7. Task 8 memory is in-process only; a process restart clears the session state. The required same-session and fresh-session transcript demonstrations are saved under `transcripts/`.
8. Task 10 groundedness validation uses lexical token overlap as a deterministic heuristic. It can miss unsupported claims or reject valid paraphrases; it is not a semantic fact-checker.
9. CrewAI deprecation warnings remain. `tracing=False` alone is not proof of zero outbound network traffic. Set telemetry-disabling environment variables before `crew.kickoff()` and independently verify behavior in the target environment.
10. Task 11 has been verified with automated and local smoke tests, but has not undergone production load testing, external security testing, authentication/authorization review, or healthcare compliance assessment.
11. Task 12 logging and PII-masking tests pass for the covered cases. The fixed-format masker does not identify all types of sensitive information. The implementation does not configure a dedicated production log destination, retention, rotation, or production log-access controls.
12. Task 13's judge is deterministic and pattern-based. Regex matches do not prove semantic entailment, and the grounding score relies on retrieval metadata rather than independently checking each answer claim against the full evidence text. Human review is required.
13. The Task 13 evaluation includes three cases where the generated response did not match every expected fact: insurance-claim escalation, lab-test timing caveats, and telemedicine confirmation details. These gaps are reflected in the recorded scores and should not be hidden by inflating the results.
14. The Task 13 score averages are outputs of the current scoring rules, not proof of production clinical accuracy, safety, or generalization.
15. Task 14's deterministic AutoGen mock only recognizes a limited set of configured refund-guarantee phrases; it is not a general semantic policy-compliance engine.
16. Part 4 Task 15 governance/budget-cap controls and Task 16 response caching remain pending.

## Remaining capstone work

The remaining requirements from the capstone objective are:

- **Part 3 — Task 13:** The 15-query evaluation has been implemented and its report structure verified. The current evaluator is a deterministic mock rubric, not a semantic LLM judge; human review is required.
- **Part 4 — Task 14:** Implemented and tested. The AutoGen review stage approves a supported draft unchanged and revises a draft containing the configured unsupported refund guarantee, with structured Pydantic verdicts. The deterministic mock's phrase coverage is limited; see the Task 14 limitations.
- **Part 4 — Task 15:** Demonstrate least-autonomy enforcement, provide a justified risk classification, and show that the runtime cost-budget cap rejects an oversized simulated request.
- **Part 4 — Task 16:** Implement in-memory response caching keyed by normalized query text and demonstrate a repeated identical query producing a cache hit that avoids a redundant LLM/tool call, with before/after evidence.

The CrewAI crew uses deterministic `MOCK_LLM` mode and extends `crewai.llms.base_llm.BaseLLM`; it does not require hosted-model API keys for inference. For strict offline acceptance, set `CREWAI_DISABLE_TELEMETRY=true` or `OTEL_SDK_DISABLED=true` before importing/running CrewAI and before calling `crew.kickoff()`. The transcript-generation script sets these variables before importing project modules.

A strict zero-network claim still requires verification with network access disabled or network activity independently monitored; a mock LLM alone does not prove that no other component attempts an outbound connection.

## Submission

The final submission is one public GitHub repository containing the dataset, knowledge-base documents, RAG core, CrewAI orchestration, AutoGen review stage, governance controls, FastAPI deployment, evaluation outputs, and transcripts demonstrating the required behavior under `MOCK_LLM` with zero API keys. Task 14 is implemented and tested; Tasks 15–16 are still pending and must be completed before the capstone can be described as complete.

All healthcare policy and appointment information in this repository is synthetic. The Task 13 report must be interpreted in light of its deterministic scoring limitations and human-review requirement.
