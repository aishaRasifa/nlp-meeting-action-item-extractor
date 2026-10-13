from pathlib import Path
import xml.etree.ElementTree as ET


# Project folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# AMI dataset folder
AMI_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ami_public_manual_1.6.2"
)

# Words XML file
WORDS_FILE = (
    AMI_FOLDER
    / "words"
    / "EN2001a.A.words.xml"
)


# Read XML
tree = ET.parse(WORDS_FILE)
root = tree.getroot()


# Missing IDs from our latest output
target_ids = [
    "EN2001a.A.words357",
    "EN2001a.A.words393",
    "EN2001a.A.words538",
    "EN2001a.A.words539",
]


print("Checking the XML for missing IDs:\n")


for element in root.iter():

    element_id = element.get(
        "{http://nite.sourceforge.net/}id"
    )

    if element_id in target_ids:

        print("FOUND")
        print("Tag:", element.tag)
        print("ID:", element_id)
        print("Text:", element.text)
        print("Attributes:", element.attrib)
        print("-" * 50)


print("\nCheck completed.")