chunking_prompt_template = """
You are an expert document/text analyst with a strong grasp of logic, english, code and general comprehension.
Split the provided document into logical sections that make sense for a Q&A system. Each section should be self-contained and cover a specific topic or concept.
<DOCUMENT>
{document}
</DOCUMENT>

Use this format:
## Section Name
Section content with all relevant details
---
## Another Section Name
Another section content
---
"""
