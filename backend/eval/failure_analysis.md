# Failure Analysis

> Documenting concrete failure cases observed during evaluation, with root-cause analysis and proposed fixes.

---

## Failure Case 1: OCR Character Confusion (`0` ↔ `O`, `l` ↔ `1`)

**Description:**
PaddleOCR frequently misreads visually similar characters in handwritten prescriptions. Numerals are confused with letters — `Am0xicillin` instead of `Amoxicillin`, `0meprazole` instead of `Omeprazole`, `Paracetam0l` instead of `Paracetamol`.

**Root Cause:**
Handwritten medical text often uses mixed casing and rushed lettering, making `O`/`0` and `l`/`1` visually indistinguishable at the pixel level. PaddleOCR's general-purpose English model is not fine-tuned on medical vocabulary.

**Observed Behavior:**
- Drug name extraction produces typo-laden names that fail exact KB lookup.
- Without fuzzy matching, grounded explanations would fail entirely.

**Proposed Fix:**
- ✅ **Implemented**: RapidFuzz fuzzy matching in KB lookup (`build_kb.py`) tolerates OCR typos with a threshold of 75%.
- 🔮 **Future**: Fine-tune TrOCR on medical vocabulary to reduce CER. A medical-specific character correction post-processor (e.g., dictionary-constrained beam search) would further improve accuracy.

---

## Failure Case 2: Multi-Line Dosage Instructions Split Across Lines

**Description:**
Some prescriptions write dosage instructions spanning two physical lines — e.g., the drug name on one line and the frequency/duration on the next. The OCR engine treats them as separate text blocks.

**Root Cause:**
PaddleOCR's bounding box detection segments by spatial proximity. When a doctor writes instructions below the drug name (instead of on the same line), the Y-coordinate gap exceeds the `line_merge_threshold`, causing them to be placed in different lines.

**Observed Behavior:**
- BioBERT NER sees fragmented input: `["Tab Amoxicillin 500mg", "1-0-1 after food 5 days"]`
- The NER model fails to associate the frequency/duration with the drug because they appear in a separate token sequence.

**Proposed Fix:**
- 🔮 **Future**: Implement a line-merging heuristic that detects orphan lines containing only frequency/duration/instruction tokens (via regex patterns like `\d-\d-\d`, `SOS`, `after food`) and merges them with the preceding drug line.
- 🔮 **Future**: Use the LLM extraction path as a secondary validator — it handles multi-line context better due to its attention mechanism.

---

## Failure Case 3: Unusual Lab Units (`lakhs/µL`) Causing Parse Errors

**Description:**
Indian lab reports commonly use `lakhs/µL` for platelet counts (e.g., `2.5 lakhs/µL`), which is non-standard internationally. The extraction pipeline sometimes misparses the value or fails to match the reference range correctly.

**Root Cause:**
The LLM extraction prompt expects standardized units, and `lakhs` is a regional convention. When the value is `2.5` in `lakhs/µL`, the actual count is `250,000 cells/µL`. Reference ranges in the KB use the `lakhs` unit, but extracted values from OCR may inconsistently include or omit the `lakhs` qualifier.

**Observed Behavior:**
- Value `2.5` with unit `lakhs/uL` is correctly identified, but `250000 cells/uL` from a different report for the same test causes a range mismatch.
- Abnormality detection gives incorrect results when units are mismatched.

**Proposed Fix:**
- ✅ **Implemented**: Lab KB (`lab_kb.json`) stores ranges in `lakhs/µL` for platelets, matching common Indian report format.
- 🔮 **Future**: Add a unit normalization layer that converts between `lakhs/µL` and `cells/µL` before range comparison. Detect the unit from context and normalize to a canonical form.

---

## Failure Case 4: Brand Name Not in KB → Ungrounded Explanation

**Description:**
Regional or less common brand names (e.g., `Zimax 500mg` for Azithromycin, `Gluconorm SR` for Metformin) are not found in the curated drug KB, triggering the ungrounded fallback explanation path.

**Root Cause:**
The drug KB contains ~1,200+ entries covering major generics and popular Indian brands, but the Indian pharmaceutical market has thousands of branded formulations. The KB cannot cover every regional brand variant.

**Observed Behavior:**
- `kb.lookup_drug("Zimax 500mg")` returns `None` because only `Azithromycin` and common brands like `Zithromax` are indexed.
- The fallback explanation is generic and tells the patient to "ask your pharmacist," which reduces user trust.

**Proposed Fix:**
- ✅ **Implemented**: Fuzzy matching with RapidFuzz catches common misspellings and close variants.
- 🔮 **Future**: Expand the brand-to-generic mapping using the CDSCO (India drug authority) database. Consider adding an LLM-powered brand resolution step before KB lookup: ask the LLM "What is the generic name for Zimax?" and then look up the generic.

---

## Failure Case 5: Hindi Translation Inconsistencies in LLM Output

**Description:**
When the LLM generates bilingual explanations, the Hindi (`explanation_hi`) output sometimes uses English medical terms transliterated into Devanagari (e.g., "एंटीबायोटिक" for "antibiotic") rather than a simpler Hindi equivalent, or mixes English and Hindi mid-sentence.

**Root Cause:**
The LLM (Gemini / Groq) doesn't have strong guardrails for Hindi purity. Medical terminology in Hindi often uses English loanwords because standardized Hindi medical terms aren't widely used in everyday conversation. The prompt says "simple Hindi" but doesn't enforce a specific vocabulary level.

**Observed Behavior:**
- Output contains code-mixed sentences: `"यह एक antibiotic है जो bacterial infection का इलाज करती है।"`
- Inconsistent across runs — sometimes fully Hindi, sometimes heavily code-mixed.

**Proposed Fix:**
- ✅ **Partially addressed**: The system prompt instructs "simple Hindi (Devanagari script)" which helps.
- 🔮 **Future**: Add a post-processing validator that checks the Hindi output for Latin characters (excluding numerals) and re-prompts the LLM if code-mixing exceeds a threshold (e.g., > 20% English words). Alternatively, use few-shot examples of pure Hindi medical explanations in the prompt.
