
from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_PATH = Path("models/all-MiniLM-L6-v2")

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

print(f"Downloading/loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

print(f"Saving model locally to: {MODEL_PATH.resolve()}")
model.save(str(MODEL_PATH))

# Verify that the saved local copy can be loaded.
local_model = SentenceTransformer(
    str(MODEL_PATH.resolve()),
    local_files_only=True,
)

vector = local_model.encode(
    "Appointment cancellation policy",
    normalize_embeddings=True,
)

print("Model saved and reloaded successfully.")
print("Embedding dimensions:", len(vector))
print("Local model verification: PASSED")
