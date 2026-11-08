# Retrieval Evaluation Results

## 1. Experimental Configuration

- **Embedding model:** `all-MiniLM-L6-v2`
- **Embedding dimensions:** 384
- **Vector database:** ChromaDB
- **Distance metric:** Cosine distance
- **Fixed-size chunking:** 350 characters, 60-character overlap
- **Sentence-based chunking:** 350-character target, 1-sentence overlap
- **Retrieval depth:** Top 3 chunks, evaluated at the parent-document level
- **In-scope queries:** 5
- **Out-of-scope queries:** 2

Retrieved chunks were mapped to their source document filenames. Duplicate source filenames were removed before calculating document-level precision and recall.

**Metrics:**

- Precision = Relevant retrieved documents / All retrieved documents
- Recall = Relevant retrieved documents / All relevant documents

## 2. Fixed-Size Chunking Results

| Query | Relevant document(s) | Retrieved parent documents | Precision calculation | Recall calculation |
|---|---|---|---|---|
| How many hours before an appointment can I cancel? | `cancellation_rescheduling.md` | `cancellation_rescheduling.md`, `consultation_fees.md` | 1/2 = 0.500 | 1/1 = 1.000 |
| What is the cardiology consultation fee range? | `consultation_fees.md` | `consultation_fees.md`, `follow_up_discount.md` | 1/2 = 0.500 | 1/1 = 1.000 |
| How long does a routine blood test take? | `lab_test_turnaround.md` | `lab_test_turnaround.md`, `telemedicine.md` | 1/2 = 0.500 | 1/1 = 1.000 |
| What conditions are required for telemedicine? | `telemedicine.md` | `second_opinion.md`, `telemedicine.md` | 1/2 = 0.500 | 1/1 = 1.000 |
| How can a patient request a second opinion? | `second_opinion.md` | `prescription_refills.md`, `second_opinion.md` | 1/2 = 0.500 | 1/1 = 1.000 |

**Mean precision calculation:**

`(0.500 + 0.500 + 0.500 + 0.500 + 0.500) / 5 = 0.500`

**Mean recall calculation:**

`(1.000 + 1.000 + 1.000 + 1.000 + 1.000) / 5 = 1.000`

## 3. Sentence-Based Chunking Results

| Query | Relevant document(s) | Retrieved parent documents | Precision calculation | Recall calculation |
|---|---|---|---|---|
| How many hours before an appointment can I cancel? | `cancellation_rescheduling.md` | `cancellation_rescheduling.md` | 1/1 = 1.000 | 1/1 = 1.000 |
| What is the cardiology consultation fee range? | `consultation_fees.md` | `consultation_fees.md` | 1/1 = 1.000 | 1/1 = 1.000 |
| How long does a routine blood test take? | `lab_test_turnaround.md` | `lab_test_turnaround.md` | 1/1 = 1.000 | 1/1 = 1.000 |
| What conditions are required for telemedicine? | `telemedicine.md` | `emergency_protocol.md`, `telemedicine.md` | 1/2 = 0.500 | 1/1 = 1.000 |
| How can a patient request a second opinion? | `second_opinion.md` | `second_opinion.md` | 1/1 = 1.000 | 1/1 = 1.000 |

**Mean precision calculation:**

`(1.000 + 1.000 + 1.000 + 0.500 + 1.000) / 5 = 0.900`

**Mean recall calculation:**

`(1.000 + 1.000 + 1.000 + 1.000 + 1.000) / 5 = 1.000`

## 4. Comparison and Numeric Recommendation

| Metric | Fixed-size chunking | Sentence-based chunking |
|---|---:|---:|
| Mean precision | 0.500 | 0.900 |
| Mean recall | 1.000 | 1.000 |
| Relevant documents retrieved | 5/5 | 5/5 |

**Recommendation:** Choose sentence-based chunking for this capstone because it achieved a mean precision of **0.900**, compared with **0.500** for fixed-size chunking, while both methods achieved **1.000 mean recall**. This is an absolute precision improvement of **0.400 (40 percentage points)** across the five tested queries; however, the sample is small, and the telemedicine query still retrieves one irrelevant document.

## 5. Similarity Threshold Calibration

The sentence-based collection produced the following top-1 cosine similarities.

| Query | Scope | Top source document | Cosine similarity |
|---|---|---|---:|
| How many hours before an appointment can I cancel? | In-scope | `cancellation_rescheduling.md` | 0.6715 |
| What is the cardiology consultation fee range? | In-scope | `consultation_fees.md` | 0.7338 |
| How long does a routine blood test take? | In-scope | `lab_test_turnaround.md` | 0.6051 |
| What conditions are required for telemedicine? | In-scope | `telemedicine.md` | 0.5815 |
| How can a patient request a second opinion? | In-scope | `second_opinion.md` | 0.8110 |
| Who won the FIFA World Cup in 2018? | Out-of-scope | `lab_test_turnaround.md` | 0.1118 |
| What is the current price of Bitcoin? | Out-of-scope | `consultation_fees.md` | 0.2236 |

- Lowest observed in-scope similarity: **0.5815**
- Highest observed out-of-scope similarity: **0.2236**

**Candidate threshold calculation:**

`(0.5815 + 0.2236) / 2 = 0.40255`

Rounded to four decimal places, the candidate threshold is **0.4026**.

The measured in-scope and out-of-scope scores are separated in this seven-query sample. The threshold is provisional and is not a guarantee that every future query will be classified correctly.

## 6. Limitations

The evaluation uses five in-scope queries and two out-of-scope queries against a synthetic capstone knowledge base. The reported metrics measure retrieval at the parent-document level, not answer correctness or completeness. Additional diverse queries and manually verified relevance labels are needed before drawing broader conclusions about retrieval performance or threshold reliability.