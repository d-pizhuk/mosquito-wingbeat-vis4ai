# GenAI Reflection – Phase I (AI-aided pair)

## What we used AI for

We used Claude (Anthropic) as an implementation tool throughout the Phase I black-box analysis. Our role was to design the investigation, define the hypotheses, interpret results, and decide what to report. The AI handled code generation and figure production so we could move faster and explore more directions in parallel.

Concretely, we used AI for:

- **Code generation:** We described analysis goals and the AI implemented them as notebook cells. This freed us from debugging boilerplate and let us focus on what the results meant.
- **Parallel analysis:** For the four deeper-analysis directions we ran multiple AI subagents simultaneously — something that would have taken days of sequential coding to do ourselves.
- **Statistical testing:** When we suspected a time-of-day effect but judged that visual inspection alone was not sufficient, we directed the AI to run formal statistical tests (Kruskal–Wallis, Mann–Whitney, Spearman). The tests came back non-significant, which we used to overrule the visual impression.
- **Presentation:** We directed the AI to assemble the figures into a structured slide deck, then reviewed and edited every slide for accuracy.

---

## What worked well

**Speed.** A full analysis pass that would have taken days of solo coding was done in a few hours. The AI handled boilerplate (data loading, axis labelling, figure export) instantly, leaving us to focus on interpretation.

**Unexpected findings surface faster.** Because the AI runs every hypothesis without ego investment, it reported the reversed H2 lift for *An. arabiensis* (atypical frequency clips were *easier*, not harder) as clearly as a confirmed finding. A human analyst might have dismissed or delayed reporting a result that contradicted the hypothesis.

**Keeping us honest.** We pushed for statistical tests when visual inspection looked suspicious, and the tests correctly returned non-significant results. Having a tool that runs tests instantly made it easier to hold ourselves to a higher standard of evidence.

**Multiple angles in parallel.** Spawning separate agents for PCA, session clustering, probability scatter, and Lorenz curves simultaneously would be impossible with a human pair in real time.

---

## What did not work well / limitations

**Hallucinated confidence.** The AI occasionally described a plot's visual appearance in a way that matched the expected finding rather than what the data actually showed. We caught this when the gambiae violin plot looked "not like a violin" — the AI initially described it as expected before we confirmed the skew was real.

**No ground truth for domain facts.** When we asked about the biological reason for the arabiensis vs gambiae frequency overlap, the AI gave a plausible explanation (sibling species share wing morphology) that we could not independently verify from the dataset alone. These causal claims should be treated as hypotheses, not findings.

**Statistical test selection is a black box.** The AI chose Kruskal–Wallis without explaining why it was preferred over ANOVA. We accepted it because the reasoning given (non-normal distributions) was plausible, but we did not independently verify the choice.

**Loss of intermediate reasoning.** When the AI runs a chain of analysis steps, it is not always clear which intermediate step drove the conclusion. For reproducibility and explanation, we had to ask for explicit reasoning rather than just accepting the final chart.

---

## How it changed how we worked

Working with AI shifted our focus from implementation to analytical thinking. Instead of spending hours writing and debugging code, we spent that time deciding which questions to ask, what the results actually meant, and which findings were worth reporting. This is a different kind of work — and in some ways harder, because a vague question produces a generic answer. Getting useful output required us to be precise about what we wanted and critical about what we got back.

We also had to actively decide what not to include. The AI will produce a chart for every hypothesis regardless of whether the result matters. We kept three null results (H2 for arabiensis, the time-of-day tests, the no-bad-day calendar plot) and cut others — that curation was entirely our decision.

---

## What we verified manually vs trusted the AI for

| Item | What we did |
|---|---|
| `f0_hz` recomputation | Spot-checked 20 clips manually; confirmed medians were anatomically plausible |
| Gini coefficients | Verified formula against a worked example from the literature |
| Statistical test results | Re-ran one test (Kruskal–Wallis) manually to confirm p-value |
| Spectrogram figure | Listened to 10 audio clips from each category to confirm visual claims |
| Session ID parsing | Checked 5 parsed timestamps against the raw folder names |
| Slide content | Reviewed every claim in the deck against the notebook output |

We did **not** independently re-derive: the PCA implementation (accepted numpy SVD as correct), the Lorenz curve algorithm, or the confidence CDF computation.

---

## Overall assessment

GenAI assistance made the Phase I analysis meaningfully richer than what a two-person team could have produced in the same time without it. The critical skill is not in using the AI but in questioning its outputs: our best findings came from moments where we disagreed with the AI's initial framing and pushed for a better test or a cleaner interpretation.

The AI is a fast, tireless analyst with no domain expertise and no sense of when a result matters. Those judgements still belong to the human pair.
