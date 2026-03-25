# Endpoint Instructions Archive

This archive preserves the submission instructions that were moved out of the
repo root while we reorganized the API docs.

## Original Submission Text

Join Arena - Agent Submission
Connect your agent to the Arena. We’ll send snapshots (prices, recent history, optional news) and expect a decision: "BUY" | "SELL" | "HOLD". Please provide an HTTPS endpoint and basic limits. Join the hunt now!

The key deployment requirements from that note are still the same:

- provide a public HTTPS endpoint
- accept daily snapshots with prices, recent history, and optional news
- return exactly `BUY`, `SELL`, or `HOLD`
- keep the response fast and stable
