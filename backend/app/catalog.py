from functools import lru_cache
from pathlib import Path
import re
import unicodedata
import xml.etree.ElementTree as ET
import zipfile


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "docs" / "BOT.xlsx"

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def _slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", ascii_value.lower()).strip("-")


def _column(reference: str) -> str:
    match = re.match(r"[A-Z]+", reference)
    return match.group(0) if match else ""


def _cell_value(cell: ET.Element, shared: list[str]) -> str:
    kind = cell.attrib.get("t")
    value_node = cell.find(f"{{{MAIN_NS}}}v")
    inline_node = cell.find(f"{{{MAIN_NS}}}is")

    if value_node is not None:
        raw = value_node.text or ""
        return shared[int(raw)] if kind == "s" and raw else raw
    if inline_node is not None:
        return "".join(
            node.text or "" for node in inline_node.iter(f"{{{MAIN_NS}}}t")
        )
    return ""


@lru_cache(maxsize=1)
def get_antenna_catalog() -> dict:
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Catalogue introuvable : {CATALOG_PATH}")

    with zipfile.ZipFile(CATALOG_PATH) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [
                "".join(
                    node.text or "" for node in item.iter(f"{{{MAIN_NS}}}t")
                )
                for item in shared_root
            ]

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(
            archive.read("xl/_rels/workbook.xml.rels")
        )
        targets = {
            relation.attrib["Id"]: relation.attrib["Target"]
            for relation in relationships.findall(
                f"{{{PKG_REL_NS}}}Relationship"
            )
        }

        sheet_path: str | None = None
        for sheet in workbook.findall(f".//{{{MAIN_NS}}}sheet"):
            if sheet.attrib.get("name") == "grand type antenne":
                relation_id = sheet.attrib[f"{{{REL_NS}}}id"]
                target = targets[relation_id].lstrip("/")
                sheet_path = target if target.startswith("xl/") else f"xl/{target}"
                break

        if not sheet_path:
            raise ValueError("La feuille 'grand type antenne' est introuvable")

        sheet_root = ET.fromstring(archive.read(sheet_path))

    families: list[dict] = []
    current_family: dict | None = None
    for row in sheet_root.findall(f".//{{{MAIN_NS}}}row"):
        row_number = int(row.attrib.get("r", "0"))
        if row_number <= 1:
            continue

        values = {
            _column(cell.attrib.get("r", "")): _cell_value(cell, shared).strip()
            for cell in row.findall(f"{{{MAIN_NS}}}c")
        }
        family_name = re.sub(r"^\d+\.\s*", "", values.get("A", "")).strip()
        variant_name = values.get("B", "").strip()

        if family_name:
            current_family = {
                "id": _slug(family_name),
                "name": family_name,
                "antennas": [],
            }
            families.append(current_family)
        if variant_name and current_family is not None:
            current_family["antennas"].append(variant_name)

    if not families:
        raise ValueError("Le catalogue ne contient aucune famille d'antenne")

    return {
        "source": "backend/docs/BOT.xlsx",
        "families": families,
        "family_count": len(families),
        "antenna_count": sum(len(family["antennas"]) for family in families),
    }


def validate_antenna_selection(family_id: str, variant: str) -> dict:
    for family in get_antenna_catalog()["families"]:
        if family["id"] == family_id and variant in family["antennas"]:
            return {"family_id": family_id, "family": family["name"], "variant": variant}
    raise ValueError("La famille ou la variante d'antenne est inconnue")
