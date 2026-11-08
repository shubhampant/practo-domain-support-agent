
EVALUATION_QUERIES = [
    {
        "query": "How many hours before an appointment can I cancel?",
        "relevant_documents": ["cancellation_rescheduling.md"],
        "scope": "in_scope",
    },
    {
        "query": "What is the cardiology consultation fee range?",
        "relevant_documents": ["consultation_fees.md"],
        "scope": "in_scope",
    },
    {
        "query": "How long does a routine blood test take?",
        "relevant_documents": ["lab_test_turnaround.md"],
        "scope": "in_scope",
    },
    {
        "query": "What conditions are required for telemedicine?",
        "relevant_documents": ["telemedicine.md"],
        "scope": "in_scope",
    },
    {
        "query": "How can a patient request a second opinion?",
        "relevant_documents": ["second_opinion.md"],
        "scope": "in_scope",
    },
]

OUT_OF_SCOPE_QUERIES = [
    "Who won the FIFA World Cup in 2018?",
    "What is the current price of Bitcoin?",
]
