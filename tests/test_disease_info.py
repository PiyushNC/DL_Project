from src.disease_info import get_disease_information, parse_class_label


def test_parse_healthy_plant_label():
    species, disease = parse_class_label("Tomato_healthy")
    assert species == "Tomato"
    assert disease == "Healthy"


def test_healthy_information_has_distinct_status():
    info = get_disease_information("Tomato_healthy")
    assert info.status == "Healthy"
    assert "diagnosis" in info.note.lower()


def test_disease_information_includes_disclaimer():
    info = get_disease_information("Apple___Apple_scab")
    assert info.status == "Diseased"
    assert "Educational" in info.note
