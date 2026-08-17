import re
from typing import List
from openai import OpenAI


class OllamaLLM:
    def __init__(
        self,
        model_name="qwen2.5:1.5b",
        host="http://127.0.0.1:11434/v1"
    ):
        self.client = OpenAI(
            base_url=host,
            api_key="ollama"
        )
        self.model_name = model_name
        print(f"\nLoaded Ollama Model : {model_name}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_context(self, contexts: List) -> str:
        if not contexts:
            return "No relevant documents found."
        formatted = []
        for i, doc in enumerate(contexts, start=1):
            content = doc.get("content", "").strip()
            metadata = doc.get("metadata", {})
            source = metadata.get("source_file", "Unknown")
            formatted.append(
                f"Document {i}\nSource: {source}\n\n{content}\n"
            )
        return "\n".join(formatted)

    def _clean_response(self, text: str) -> str:
        """
        Post-process the LLM response to guarantee no markdown symbols
        or commas reach the mobile UI no matter what the model outputs.

        Steps applied in order:
          1. Strip heading markers  (#, ##, ###, ...)
          2. Convert dash/star/plus list markers to • bullets
          3. Remove all remaining forbidden characters
          4. Remove commas
          5. Collapse extra whitespace
        """
        lines = text.splitlines()
        cleaned = []

        for line in lines:
            line = line.strip()

            # 1. Remove markdown heading markers (e.g. ## Heading)
            line = re.sub(r'^#{1,6}\s*', '', line)

            # 2. Convert leading dash/star/plus list markers to • bullet
            #    Only matches a dash/star/plus followed by a space at line start
            #    so hyphens inside words (disease-free, copper-based) are kept.
            line = re.sub(r'^[-*+]\s+', '\u2022 ', line)

            # 3. Strip forbidden markdown symbols wherever they appear.
            #    NOTE: do NOT include '-' here — it is already handled above
            #    for line-start markers and must be preserved inside words.
            line = re.sub(r'[#*_/\\~`|<>\[\]{}()]', '', line)

            # 4. Remove commas
            line = line.replace(',', '')

            # 5. Collapse multiple spaces that may result from removals
            line = re.sub(r'  +', ' ', line).strip()

            cleaned.append(line)

        # Remove consecutive blank lines (keep at most one)
        result_lines = []
        prev_blank = False
        for line in cleaned:
            is_blank = (line == '')
            if is_blank and prev_blank:
                continue
            result_lines.append(line)
            prev_blank = is_blank

        return "\n".join(result_lines).strip()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_answer(self, query: str, contexts: List) -> str:
        formatted_context = self._format_context(contexts)

        system_message = (
            "You are a professional agricultural expert assisting farmers via a mobile app. "
            "You must produce ONLY plain text output. "
            "ABSOLUTE RULES you must never break:\n"
            "  - Never use any of these characters: # * _ / \\ ~ ` | [ ] { } ( ) = % @\n"
            "  - Never use dashes or hyphens at the start of a line.\n"
            "  - Never use commas anywhere.\n"
            "  - Never use markdown formatting of any kind.\n"
            "  - Use the bullet character \u2022 to start every list item.\n"
            "  - Write section headings as plain words on their own line with no punctuation."
        )

        prompt = (
            "Answer ONLY using the information in the documents below.\n"
            "If the answer is not found in the documents reply with exactly:\n"
            "I could not find this information in the indexed documents.\n\n"
            "OUTPUT FORMAT\n"
            "Use these section headings as plain text followed by bullet points.\n"
            "Skip any section that has no information in the documents.\n\n"
            "Scientific Name\n"
            "\u2022 write the scientific name here\n\n"
            "Symptoms\n"
            "\u2022 symptom one\n"
            "\u2022 symptom two\n\n"
            "Cause\n"
            "\u2022 cause description\n\n"
            "Favorable Conditions\n"
            "\u2022 condition one\n\n"
            "Treatment\n"
            "\u2022 treatment step\n\n"
            "Chemical Control\n"
            "\u2022 chemical option\n\n"
            "Organic Control\n"
            "\u2022 organic option\n\n"
            "Prevention\n"
            "\u2022 prevention tip\n\n"
            "DOCUMENTS\n"
            f"{formatted_context}\n\n"
            "QUESTION\n"
            f"{query}\n\n"
            "ANSWER"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )
            raw = response.choices[0].message.content.strip()
            return self._clean_response(raw)
        except Exception as e:
            print(f"\nOllama Error : {e}")
            return "Sorry I could not generate an answer."

    def test_connection(self) -> bool:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Say Hello"}]
            )
            print("\nOllama Connected Successfully\n")
            print(response.choices[0].message.content)
            return True
        except Exception as e:
            print(e)
            return False
