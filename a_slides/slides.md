# UOC Slide Deck — Claude for PowerPoint Instructions

---

## Slide 1 — Title
*Leave as is.*

---

## Slide 2 — Literature Review
*Leave as is.*

---

## Slide 3 — How Does a Model Behave?

**Slide title:** "How Does a Model Behave?"

**Bullet points** (placed below the title, above the grid — concise, left-aligned):

- A model's response falls into one of five behaviours: four defined by two axes (*is the input answerable?* × *commit or abstain?*), plus general utility (E) for everyday tasks outside the answerability setting.
- **Only A is the problem** — confidently answering when it shouldn't.
- Goal: eliminate A, protect B and C, preserve E, and never push the model toward D.

---

**Layout overview:**
- A 2×2 grid of quadrant boxes dominates the slide, spanning the upper ~75% of the content area.
- Below the grid, a single wide 5th box spans the full width, representing General Utility.
- Column headers sit above the grid: **COMMIT** (left) and **ABSTAIN** (right).
- Row headers sit to the left of the grid: **UNANSWERABLE** (top row) and **ANSWERABLE** (bottom row).

---

### Column and row headers

- **COMMIT** header (above left column): bold, dark text.
- **ABSTAIN** header (above right column): bold, dark text.
- **UNANSWERABLE** row label (left of top row): bold, rotated or inline, dark text.
- **ANSWERABLE** row label (left of bottom row): bold, rotated or inline, dark text.

---

### Quadrant cells (2×2)

Each cell contains:
1. A **behaviour label** tag (letter + name) in the top-left corner of the cell.
2. A **✓ or ✗** indicator in the top-right corner.
3. A short **example exchange** (Q + model response) in the body.

---

#### Top-left — UNANSWERABLE × COMMIT → Over-commitment ✗

- **Border / accent colour:** Red (`#C0392B`), light red fill (`#FDEDEC`).
- **Label tag:** **A — Over-commitment**
- **Indicator:** ✗ (red cross)
- **Note:** This is the **unlearning target** — visually emphasise this cell (thicker border or subtle highlight badge reading "unlearn target").
- **Example:**
  - Q: *"Which language is the most popular in the continent?"*
  - A: *"The most popular language in Africa is Swahili…"*

---

#### Top-right — UNANSWERABLE × ABSTAIN → Legitimate Abstention ✓

- **Border / accent colour:** Green (`#1E8449`), light green fill (`#EAFAF1`).
- **Label tag:** **B — Legitimate Abstention**
- **Indicator:** ✓ (green checkmark)
- **Example:**
  - Q: *"Which language is the most popular in the continent?"*
  - A: *"I do not have enough information to answer that."*

---

#### Bottom-left — ANSWERABLE × COMMIT → Legitimate Commitment ✓

- **Border / accent colour:** Blue (`#1060C0`), light blue fill (`#EBF5FB`).
- **Label tag:** **C — Legitimate Commitment**
- **Indicator:** ✓ (green checkmark)
- **Example:**
  - Q: *"What is the capital of France?"*
  - A: *"Paris."*

---

#### Bottom-right — ANSWERABLE × ABSTAIN → Over-abstention ✗

- **Border / accent colour:** Orange-red (`#E65000`), light orange fill (`#FFF3E0`).
- **Label tag:** **D — Over-abstention**
- **Indicator:** ✗ (red cross)
- **Example:**
  - Q: *"What is the capital of France?"*
  - A: *"I don't have enough information to answer that."*

---

### 5th box — General Utility (full width, below the 2×2 grid)

- **Border / accent colour:** Grey (`#888888`), very light grey fill.
- **Label tag:** **E — General Utility**
- **Indicator:** ✓ (green checkmark)
- **Description text:** "Ordinary instruction-following outside the answerability setting."
- **Example:**
  - Q: *"Write a poem about rain."*
  - A: *"Drops on the window…"*

---

### Visual style notes

- Consistent rounded-corner boxes throughout.
- The **A — Over-commitment** cell (bottom-left) should be the most visually prominent — thicker red border and/or a small badge/ribbon in the corner saying **"unlearn"**.
- ✓ indicators in green, ✗ indicators in red.
- Keep font sizes large enough to be readable on a projected slide; example text can be italic.
- Overall colour palette matches the project figures: red for commit/bad, green for abstain/good, blue for legit-commit, grey for utility.
