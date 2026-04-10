{{editorial_prompt}}

You are in the final ranking stage for a podcast trailer workflow.
The candidates below were already shortlisted from the full transcript.
Choose exactly {{selection_count}} candidates.
Rank them from strongest to weakest.
Optimize for the best trailer moments in context and avoid redundant picks when a more diverse set of strong moments is available.
Do not invent new ranges and do not alter the sentence boundaries.

Return JSON only with this schema:

```json
{
  "selections": [
    {
      "candidate_id": "C001",
      "title": "Short title",
      "rationale": "Why this is a top trailer moment."
    }
  ]
}
```

Candidate shortlist:

{{candidate_shortlist}}
