{{editorial_prompt}}

You are in the first-pass nomination stage for a podcast trailer workflow.
This is transcript chunk {{chunk_index}} of {{chunk_total}} from the full episode.

Your job in this stage is to identify the strongest candidate moments from this chunk only.
Each candidate must be a continuous selection of 1 to {{max_selection_sentences}} consecutive sentences.
Favor preserving setup and payoff over trimming too tightly.
Do not return clipped fragments or mid-thought excerpts.
Different candidates should be meaningfully distinct and should not heavily overlap unless the moment is exceptional.

Return up to {{candidate_limit}} candidates as JSON only with this schema:

```json
{
  "candidates": [
    {
      "start_id": "S0001",
      "end_id": "S0003",
      "title": "Short title",
      "scores": {
        "curiosity_gap": 0,
        "emotional_peak": 0,
        "bold_claim": 0,
        "vivid_storytelling": 0,
        "high_stakes_setup": 0,
        "quotable_shareable": 0
      },
      "total_score": 0,
      "rationale": "Why this range belongs on the shortlist."
    }
  ]
}
```

Transcript chunk:
{{transcript_chunk}}
