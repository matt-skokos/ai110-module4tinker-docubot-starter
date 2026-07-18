"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

import os
import glob
import string
from collections import Counter

class DocuBot:
    STOPWORDS = {
        "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or",
        "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "how", "what", "when", "where", "why", "which",
        "i", "you", "it", "this", "that", "with", "as", "can", "will",
    }

    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load documents into memory
        self.documents = self.load_documents()  # List of (filename, text)

        # Split documents into section-level chunks for retrieval
        self.chunks = self.chunk_documents(self.documents)  # List of (filename, chunk_text)

        # Build a retrieval index (implemented in Phase 1)
        self.index = self.build_index(self.chunks)

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder.
        Returns a list of tuples: (filename, text)
        """
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)
                docs.append((filename, text))
        return docs

    # -----------------------------------------------------------
    # Chunking
    # -----------------------------------------------------------

    def _split_on_heading(self, text, level):
        """
        Splits text into pieces, each starting at a line that begins a
        markdown heading of the given level (e.g. level=2 -> "## ").
        Content before the first matching heading becomes its own leading piece.
        """
        marker = "#" * level + " "
        lines = text.splitlines(keepends=True)

        pieces = []
        current = []
        for line in lines:
            if line.startswith(marker) and current:
                pieces.append("".join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            pieces.append("".join(current))
        return pieces

    def _has_heading(self, text, level):
        marker = "#" * level + " "
        return any(line.startswith(marker) for line in text.splitlines())

    def chunk_documents(self, documents):
        """
        Splits each document into section-level chunks so retrieval returns
        focused snippets instead of whole files.

        - Splits on "## " headings; each top-level section becomes a chunk
          (any intro text before the first "## " becomes its own chunk).
        - If a section itself contains "### " subheadings (e.g. a "Tables"
          section listing several tables), it is split further so each
          subsection becomes its own chunk, prefixed with the parent "## "
          heading for context.

        Returns a list of (filename, chunk_text) tuples.
        """
        chunks = []
        for filename, text in documents:
            for section in self._split_on_heading(text, level=2):
                if self._has_heading(section, level=3):
                    section_heading = section.splitlines()[0]
                    for subsection in self._split_on_heading(section, level=3):
                        if not subsection.strip().startswith("###"):
                            # Leading fragment before the first "### " is just
                            # the parent heading itself; already captured above.
                            continue
                        subsection = f"{section_heading}\n\n{subsection.strip()}"
                        chunks.append((filename, subsection))
                elif section.strip():
                    chunks.append((filename, section.strip()))
        return chunks

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    def _tokenize(self, text):
        """
        Splits text into lowercase words with surrounding punctuation stripped.
        """
        words = text.lower().split()
        tokens = [word.strip(string.punctuation) for word in words]
        return [token for token in tokens if token]

    def _query_tokens(self, query):
        """
        Tokenizes a query and drops common "fluff" words that carry little
        search value (see STOPWORDS), so they can't inflate relevance scores.
        """
        return {token for token in self._tokenize(query) if token not in self.STOPWORDS}

    def build_index(self, documents):
        """
        Build a tiny inverted index mapping lowercase words to the documents
        they appear in.

        Example structure:
        {
            "token": ["AUTH.md", "API_REFERENCE.md"],
            "database": ["DATABASE.md"]
        }

        Keep this simple: split on whitespace, lowercase tokens,
        ignore punctuation if needed.
        """
        index = {}
        for filename, text in documents:
            for token in set(self._tokenize(text)):
                index.setdefault(token, []).append(filename)
        return index

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def score_document(self, query, text):
        """
        Return a simple relevance score for how well the text matches the query.

        Suggested baseline:
        - Convert query into lowercase words
        - Count how many appear in the text
        - Return the count as the score
        """
        query_tokens = self._query_tokens(query)
        text_counts = Counter(self._tokenize(text))
        return sum(text_counts[token] for token in query_tokens)

    def retrieve(self, query, top_k=3):
        """
        Use the index and scoring function to select top_k relevant document snippets.

        Return a list of (filename, text) sorted by score descending.
        """
        query_tokens = self._query_tokens(query)

        candidate_filenames = set()
        for token in query_tokens:
            candidate_filenames.update(self.index.get(token, []))

        # Require at least this many distinct query words to actually appear in
        # a chunk, so one repeated word can't fake relevance for the rest of a
        # multi-word query. Short queries just need their one word to match.
        min_coverage = min(2, len(query_tokens))

        scored = []
        for filename, text in self.chunks:
            if filename not in candidate_filenames:
                continue
            score = self.score_document(query, text)
            if score == 0:
                continue
            matched_tokens = query_tokens & set(self._tokenize(text))
            if len(matched_tokens) < min_coverage:
                continue
            scored.append((score, filename, text))

        scored.sort(key=lambda item: item[0], reverse=True)
        results = [(filename, text) for _, filename, text in scored]
        return results[:top_k]

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        """
        Phase 1 retrieval only mode.
        Returns raw snippets and filenames with no LLM involved.
        """
        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        formatted = []
        for filename, text in snippets:
            formatted.append(f"[{filename}]\n{text}\n")

        return "\n---\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode.
        Uses student retrieval to select snippets, then asks Gemini
        to generate an answer using only those snippets.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        return self.llm_client.answer_from_snippets(query, snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive 'generation only' baselines.
        """
        return "\n\n".join(text for _, text in self.documents)
