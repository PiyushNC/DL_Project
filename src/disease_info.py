from dataclasses import dataclass
import re


@dataclass(frozen=True)
class DiseaseInformation:
    label: str
    species: str
    disease: str
    status: str
    symptoms: str
    prevention: str
    note: str


def parse_class_label(label: str) -> tuple[str, str]:
    """Separate common PlantVillage labels into species and disease text."""
    cleaned = re.sub(r"_+", " ", str(label)).strip()
    if " " not in cleaned:
        return "Unknown plant", cleaned

    words = cleaned.split()
    if "healthy" in cleaned.lower():
        return " ".join(words[:-1]), "Healthy"

    known_first_words = {"apple", "blueberry", "cherry", "corn", "grape",
                         "orange", "peach", "pepper", "potato", "raspberry",
                         "soybean", "squash", "strawberry", "tomato"}
    if words[0].lower() in known_first_words:
        species = words[0]
        disease = " ".join(words[1:])
        return species.title(), disease.title()

    return "Unknown plant", cleaned


def get_disease_information(label: str) -> DiseaseInformation:
    species, disease = parse_class_label(label)
    is_healthy = disease.lower() == "healthy"

    if is_healthy:
        symptoms = (
            "No disease pattern was predicted. Normal leaves may still vary in "
            "color, age, and lighting, so this is not a biological diagnosis."
        )
        prevention = (
            "Maintain suitable watering, spacing, sanitation, and routine crop "
            "monitoring. Seek a local plant specialist if physical symptoms appear."
        )
    else:
        symptoms = (
            f"The model associated this leaf with a {disease.lower()} pattern for "
            f"{species}. Typical visual symptoms can include spots, discoloration, "
            "lesions, or abnormal leaf texture. The exact symptoms vary by crop "
            "and disease."
        )
        prevention = (
            "Remove severely affected tissue if appropriate, avoid overhead watering, "
            "improve airflow, use clean tools, rotate crops, and follow local "
            "integrated disease-management guidance."
        )

    return DiseaseInformation(
        label=label,
        species=species,
        disease=disease,
        status="Healthy" if is_healthy else "Diseased",
        symptoms=symptoms,
        prevention=prevention,
        note=(
            "Educational model output only. This page does not provide a confirmed "
            "diagnosis or treatment recommendation. Consult a qualified agricultural "
            "professional before applying pesticides or other interventions."
        ),
    )
