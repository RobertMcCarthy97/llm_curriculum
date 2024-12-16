import xml.etree.ElementTree as ET


def pretty_print(element_tree: ET.ElementTree, indent: int = 0):

    print("--" * indent + element_tree.getroot().attrib["name"])
    for child in element_tree.getroot():
        pretty_print(ET.ElementTree(child), indent + 1)


if __name__ == "__main__":
    with open("llm_curriculum/prompts/txt/response.txt", "r") as f:
        response = f.read()

    etree = ET.ElementTree(ET.fromstring(response))
    pretty_print(etree)
