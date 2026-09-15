import re
import sys
from collections import OrderedDict
from docx import Document

doc = Document(sys.argv[1])
sections = OrderedDict()
current = "Front matter"
sections[current] = []
for paragraph in doc.paragraphs:
    text = paragraph.text.strip()
    if not text:
        continue
    style = paragraph.style.name if paragraph.style else ""
    if style.startswith("Heading 1"):
        current = text
        sections.setdefault(current, [])
    else:
        sections[current].append(text)

def count(texts):
    return len(re.findall(r"\b[\w]+(?:[’'-][\w]+)*\b", " ".join(texts), flags=re.UNICODE))

for heading, texts in sections.items():
    print(f"{count(texts):5d}\t{heading}")
