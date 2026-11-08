# Practo Domain Support Agent (CrewAI)

**Capstone track: Practo (Healthcare).**
This repository implements a local-first support-agent prototype for answering synthetic clinic-policy questions and looking up synthetic appointment records. **Part 1 — Tasks 1–5** and **Part 2 — Tasks 6–10** have implementation and automated-test evidence. **Task 8 is complete, including the required same-session and fresh-session transcript demonstrations.** Tasks 11–16 remain pending until their code, tests, and required demonstrations are added.

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
- [Test suite](#test-suite)
- [Current evaluation summary](#current-evaluation-summary)
- [Known limitations](#known-limitations)
- [Remaining capstone work](#remaining-capstone-work)

## Project status

| Capstone area | Status | Evidence |
|---|---|---|
| Part 1, Task 1 — seeded appointment dataset | Complete | `dataset.py`, `tests/test_dataset.py` |
| Part 1, Task 2 — 12-topic knowledge base | Complete | `data/knowledge_base/`, `tests/test_knowledge_base.py` |
| Part 1, Task 3 — two chunking/indexing strategies | Implemented | `chunking.py`, two ChromaDB collections |
| Part 1, Task 4 — grounded generation and fallback | Complete | `grounded_generator.py`, `retrieval.py`, `transcripts/task4_grounded_generation.txt` |
| Part 1, Task 5 — document-level precision/recall comparison | Complete | `run_evaluation.py`, `evaluation_results.md` |
| Part 2, Task 6 — appointment lookup and escalation score | Implemented and tested | `appointment_lookup.py`, `tests/test_appointment_lookup.py`; 11 tests pass |
| Part 2, Task 7 — CrewAI multi-agent orchestration | Implemented and tested | `crewai_support.py`, `tests/test_crewai_support.py`; deterministic mock LLM; 3 agents and `.kickoff()` |
| Part 2, Task 8 — session memory | Complete: implementation, tests, and transcript demonstrations verified | `session_memory.py`, `tests/test_session_memory.py`, `tests/test_crewai_session_memory.py`, `transcripts/task8_same_session.txt`, `transcripts/task8_fresh_session.txt` |
| Part 2, Task 9 — structured response schema | Implemented and tested | `response_schema.py`, `tests/test_response_schema.py`, `tests/test_crewai_structured_response.py` |
| Part 2, Task 10 — guardrails | Implemented and tested | `guardrails.py`, `tests/test_guardrails.py`, `tests/test_crewai_guardrails.py`, `tests/test_grounded_generator.py` |
| Part 3 — FastAPI, structured logs, 15-query evaluation | Not yet implemented/verified | To be documented after implementation |
| Part 4 — AutoGen review, governance, budget, caching | Not yet implemented/verified | To be documented after implementation |

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
                            transcript + evaluation
```

The recommended retrieval strategy based on the current small benchmark is sentence-based chunking. It achieved higher mean document-level precision while retaining the same recall as fixed-size chunking.

## Environment and reproducibility

The development environment used for the verified runs was:

- Windows with PowerShell (VS Code terminal).
- Python **3.11.9** inside a project virtual environment at `.venv`.
- SentenceTransformers with a local `all-MiniLM-L6-v2` model.
- ChromaDB for local vector storage.
- Pytest for automated tests.

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

If PowerShell says the virtual environment does not exist, create it using the Python 3.11 installation and install the dependencies from the repository's dependency file, if present:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The commands above are for environment setup. They may need network access if packages are not already cached or installed. **Runtime model inference should use the locally available embedding model and should not require an API key.** For a fully offline machine, prepare dependencies and the embedding model in advance, transfer them through the approved process, and verify runtime behavior with network access disabled.

### 2. Run the automated tests

```powershell
python -m pytest -v
```

Latest verified result from the development session: **95 passed, 0 failed**. The run completed in 30.91 seconds and reported 506 CrewAI deprecation warnings. Warnings are not test failures, but should be reviewed when upgrading CrewAI.

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

Latest verified complete-suite result: `95 passed, 0 failed` (506 CrewAI deprecation warnings).

To reproduce the escalation count for the current deterministic dataset:

```powershell
python -c "from dataset import APPOINTMENTS; from appointment_lookup import check_appointment_status; results = [check_appointment_status(a['record_id']) for a in APPOINTMENTS]; escalated = [r for r in results if r['escalation_recommended']]; print('Total appointments:', len(results)); print('Escalation threshold: 0.8'); print('Escalated:', len(escalated)); print('Not escalated:', len(results) - len(escalated)); [print(r['record_id'], '| score:', r['escalation_score'], '| status:', r['status']) for r in escalated]"
```

Verified result for the current seed-42 dataset: **7 of 50 records (14%)** are flagged for operational escalation and 43 are not. The seven IDs are `APT-0010`, `APT-0011`, `APT-0020`, `APT-0022`, `APT-0023`, `APT-0024`, and `APT-0050`. These counts depend on the dataset and scoring rule remaining unchanged.

The corresponding tests are in `tests/test_appointment_lookup.py` and cover score boundaries, score range, lookup success, case/whitespace normalization, unknown IDs, and invalid inputs.

## Part 2 — Task 7: CrewAI multi-agent orchestration

`crewai_support.py` builds a sequential CrewAI crew with three agents: Retrieval Agent, Lookup Agent, and Response Composer. The Retrieval Agent has access to the RAG lookup tool; the Lookup Agent has access to the appointment-status lookup tool. The crew runs through `.kickoff()` and uses the project's deterministic `DeterministicMockLLM` rather than a hosted inference API.

Tool invocations and tool results are captured for test/audit inspection. The crew is configured with `tracing=False` and `memory=False`; session history is handled separately by the project's in-process memory implementation. **Do not treat these settings alone as proof of zero network traffic.** CrewAI telemetry must be disabled before kickoff and runtime network behavior verified for a strict offline claim.

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

- **Fixed-format contact-number masking:** Indian mobile numbers, optionally prefixed by `+91`, are replaced with `[MASKED_CONTACT_NUMBER]` before RAG processing and session storage. This masker is not intended to identify arbitrary names, diagnoses, or insurance IDs in free text.
- **Prompt-injection detection:** suspicious instruction-override requests are rejected before CrewAI execution; the rejected raw query is not added to session history.
- **Output groundedness check:** generated answer sentences are checked against retrieved evidence. Unsupported claims cause `GroundedSupportAgent.ask()` to return `insufficient_evidence` and a refusal message.

The groundedness check is a deterministic lexical-overlap heuristic, not a semantic fact-checker and not a guarantee against all hallucinations. Use only fabricated test examples; never use real patient information.

Verify the deliberate guardrail test cases and integration:

```powershell
python -m pytest tests/test_guardrails.py tests/test_crewai_guardrails.py tests/test_grounded_generator.py -v
```

The tests demonstrate contact masking, injection rejection before crew execution, and rejection of a deliberately unsupported generated answer.

## Test suite

Run the complete test suite from the repository root:

```powershell
python -m pytest -v
```

Latest verified run:

```text
95 passed, 506 warnings in 30.91s
```

Dependency consistency check:

```powershell
python -m pip check
```

Latest verified result:

```text
No broken requirements found.
```

The tests cover dataset generation, chunking, knowledge-base structure, retrieval behavior, grounded generation, appointment lookup, CrewAI orchestration, session memory, structured responses, and guardrails. Passing tests do not by themselves prove every objective requirement or replace required demonstration transcripts.

## Current evaluation summary

- Dataset: 50 records; 5 categories; 5 statuses; 22% follow-up-required records.
- Knowledge base: 12 required topic documents.
- Chunking: 26 fixed-size chunks and 36 sentence-based chunks in separate collections.
- Embeddings: local `all-MiniLM-L6-v2`, 384 dimensions.
- Threshold: 0.4026, empirically selected from five in-scope and two out-of-scope query measurements.
- Grounded generation: five in-scope demonstrations and an out-of-scope fallback.
- Retrieval benchmark: fixed-size precision/recall = 0.500/1.000; sentence-based = 0.900/1.000.
- Task 6 escalation rule: threshold 0.8; 7/50 synthetic appointments (14%) flagged in the current dataset.
- Automated tests: 95 passed, 0 failed in the latest reported run; 506 CrewAI deprecation warnings were reported.

## Known limitations

1. The policy documents and appointment dataset are synthetic and do not represent official Practo data or policies.
2. Similarity-threshold calibration uses only seven measured queries; additional varied in-scope and out-of-scope examples are needed.
3. The retrieval evaluation contains five in-scope questions, so its precision/recall estimates should not be treated as broad performance guarantees.
4. Retrieval metrics do not measure the correctness, completeness, or safety of every generated answer.
5. The embedding model and dependencies must be prepared locally before running in a completely offline environment.
6. Task 6 escalation is a synthetic operational rule, not a clinically validated measure.
7. Task 8 memory is in-process only; process restart clears the session state. The required same-session and fresh-session transcript demonstrations are saved under `transcripts/`.
8. Task 10 groundedness validation uses lexical token overlap as a deterministic heuristic. It can miss unsupported claims or reject valid paraphrases; it is not a semantic fact-checker.
9. CrewAI deprecation warnings remain. `tracing=False` is configured, but that alone is not proof of zero outbound network traffic. Set telemetry-disabling environment variables before `crew.kickoff()` and verify behavior in the target environment.

## Remaining capstone work

The remaining requirements from the capstone objective are:

- **Part 2 — Tasks 6–10:** implementation and test evidence are present; Task 8 also has the required same-session and fresh-session transcripts.
- **Part 3 — Tasks 11–13:** at least two FastAPI HTTP endpoints and a WebSocket endpoint; structured JSONL logs with trace IDs and masked request text; 15-query evaluation reporting Accuracy, Grounding, Completeness, and Safety per query and as averages.
- **Part 4 — Tasks 14–16:** two-agent AutoGen review stage with approval and revision demonstrations; least-autonomy governance, risk classification, and runtime budget cap; response-cache hit demonstration.

The CrewAI crew uses deterministic `MOCK_LLM` mode and extends `crewai.llms.base_llm.BaseLLM`; it does not require hosted-model API keys for inference. For strict offline acceptance, set `CREWAI_DISABLE_TELEMETRY=true` (or `OTEL_SDK_DISABLED=true`) before importing/running CrewAI and before calling `crew.kickoff()`. The transcript-generation script sets these variables before importing project modules. A strict zero-network claim still requires verification with network access disabled or network activity independently monitored; a mock LLM alone does not prove that no other component attempts an outbound connection.

## Submission

The final submission is one public GitHub repository containing the dataset, knowledge-base documents, RAG core, CrewAI orchestration, review stage, governance controls, FastAPI deployment, evaluation outputs, and transcripts demonstrating the required behavior under `MOCK_LLM` with zero API keys.
