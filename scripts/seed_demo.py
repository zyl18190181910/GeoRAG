from pathlib import Path

from geo_rag.config import Settings
from geo_rag.loaders import extract_sections_from_bytes
from geo_rag.service import RagService


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    settings = Settings(storage_path=ROOT / "data" / "storage" / "knowledge.json")
    service = RagService(settings)
    for path in sorted((ROOT / "data" / "sample_docs").glob("*.md")):
        sections = extract_sections_from_bytes(path.name, path.read_bytes())
        added = service.ingest_sections(
            path.stem,
            [(section.text, section.metadata) for section in sections],
            metadata={"filename": path.name},
        )
        print(f"{path.name}: {added} chunks")
    print(f"seeded {len(service.source_ids)} sources and {len(service.chunks)} chunks")


if __name__ == "__main__":
    main()

