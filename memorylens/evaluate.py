"""
MemoryLens - Evaluation Script (LLM-as-judge)
===============================================
Runs structured benchmark suite through all three variants.
All metrics scored automatically by LLM-as-judge.

Variants:
  V0 (plain): chunk retrieval, NO memory
  VA (chunks): chunk retrieval + AgentState memory + history
  VB (episodes): episodic retrieval + AgentState memory + history

Comparison story:
  V0 vs VA: does memory + history help on top of retrieval?
  VA vs VB: does episodic structure specifically help? (core hypothesis)

Metrics:
  1. Constraint Satisfaction Rate (CSR)      — LLM-as-judge, per turn
  2. Answer Relevance Score (1-5)            — LLM-as-judge, final turn
  3. Keyword Match Rate                      — auto keyword check
  4. Response Latency (seconds)              — timed per turn

Test Set: 24 queries — 6 per family
  More queries per family = more reliable averages, less variance
  from individual LLM scoring inconsistencies

Results:
  evaluation/results.json          full results
  evaluation/results_summary.csv   comparison table for report
  evaluation/csr_per_turn.csv      CSR across turns (key finding)

Usage:
    python evaluate.py                 # all 3 variants
    python evaluate.py --mode chunks   # one variant only
"""

import json
import csv
import argparse
from pathlib import Path
import re
from langchain_ollama import OllamaLLM

from chatbot import build_graph, build_initial_state, run_turn

# ── Config ────────────────────────────────────────────────────────────────────
MODEL      = "llava"
OUTPUT_DIR = Path("evaluation")
OUTPUT_DIR.mkdir(exist_ok=True)

llm = OllamaLLM(model=MODEL)

VARIANT_LABELS = {
    "plain":    "Variant 0 (chunks, no memory)",
    "chunks":   "Variant A (chunks + memory)",
    "episodes": "Variant B (episodes + memory)"
}

# ── Benchmark Test Set ────────────────────────────────────────────────────────
# 24 queries — 6 per family
# More queries per family reduces variance from individual LLM scoring noise
# and produces more reliable averages for the comparison table

TEST_SET = [

    # ── Factual (F1-F6) ───────────────────────────────────────────────────────
    {
        "id": "F1",
        "family": "factual",
        "turns": ["How much did I spend per day in Bali?"],
        "constraints": [],
        "expected_keys": ["45", "AUD", "Bali"],
        "context_hint": "Answer should mention approx $45 AUD per day from Bali notes."
    },
    {
        "id": "F2",
        "family": "factual",
        "turns": ["What fort did I visit in Jodhpur?"],
        "constraints": [],
        "expected_keys": ["Mehrangarh"],
        "context_hint": "Answer should mention Mehrangarh Fort and the blue city view."
    },
    {
        "id": "F3",
        "family": "factual",
        "turns": ["How long did the Kelingking Beach hike take?"],
        "constraints": [],
        "expected_keys": ["45", "hour"],
        "context_hint": "45 minutes down, 1 hour back up."
    },
    {
        "id": "F4",
        "family": "factual",
        "turns": ["How did I get to Jodhpur?"],
        "constraints": [],
        "expected_keys": ["bus", "Jodhpur"],
        "context_hint": "Answer should mention taking a bus to Jodhpur."
    },
    {
        "id": "F5",
        "family": "factual",
        "turns": ["How long was the Marina walk in Dubai?"],
        "constraints": [],
        "expected_keys": ["7", "Dubai"],
        "context_hint": "About 7 kilometers."
    },
    {
        "id": "F6",
        "family": "factual",
        "turns": ["What was the highlight experience of my Dubai trip?"],
        "constraints": [],
        "expected_keys": ["desert", "safari"],
        "context_hint": "The desert safari — dune bashing, sunset, stars above the camp."
    },

    # ── Cross-Modal (C1-C6) ───────────────────────────────────────────────────
    {
        "id": "C1",
        "family": "cross_modal",
        "turns": ["Find me a photo I took of a temple in Bali"],
        "constraints": [],
        "expected_keys": ["temple", "Bali"],
        "context_hint": "Should return Tirtha Empul or Pura Besakih photo description."
    },
    {
        "id": "C2",
        "family": "cross_modal",
        "turns": ["Which destination had the most dramatic landscape based on my photos?"],
        "constraints": [],
        "expected_keys": ["Nusa Penida", "cliff"],
        "context_hint": "Nusa Penida Kelingking cliff — most dramatic visual."
    },
    {
        "id": "C3",
        "family": "cross_modal",
        "turns": ["Show me a photo that I took from any fort in Jodhpur"],
        "constraints": [],
        "expected_keys": ["Jodhpur", "mehrangarh"],
        "context_hint": "mehrangarh fort situated at the top of the hill."
    },
    {
        "id": "C4",
        "family": "cross_modal",
        "turns": ["Find a photo I took that shows water — ocean, waterfall, or pool"],
        "constraints": [],
        "expected_keys": ["water", "beach"],
        "context_hint": "Could be Nusa Penida billabong, Kelingking beach, Dudhsagar falls, or Agonda beach."
    },
    {
        "id": "C5",
        "family": "cross_modal",
        "turns": ["Based on my photos which destination looked most spiritual or cultural?"],
        "constraints": [],
        "expected_keys": ["temple", "Bali"],
        "context_hint": "Bali has multiple temple photos — Tirtha Empul, Pura Besakih, Tanah Lot. India has Jama Masjid."
    },
    {
        "id": "C6",
        "family": "cross_modal",
        "turns": ["Find a photo from my India trip — what does it show?"],
        "constraints": [],
        "expected_keys": ["India", "fort"],
        "context_hint": "Red Fort Delhi, Amber Fort Jaipur, Mehrangarh Fort Jodhpur, or Jama Masjid."
    },

    # ── Multi-Hop (M1-M6) ─────────────────────────────────────────────────────
    {
        "id": "M1",
        "family": "multi_hop",
        "turns": ["Which of my trips had the best food to cost ratio?"],
        "constraints": [],
        "expected_keys": ["Bali", "India", "cost"],
        "context_hint": "Compare Bali $45/day, India $55/day, Singapore $130/day, Dubai $180/day."
    },
    {
        "id": "M2",
        "family": "multi_hop",
        "turns": ["Comparing all my trips which destination was required the most physical effort?"],
        "constraints": [],
        "expected_keys": [ "Nusa Penida"],
        "context_hint": "Kelingking hike in Nusa Penida or India heat and walking."
    },
    {
        "id": "M3",
        "family": "multi_hop",
        "turns": ["If I had a budget of ONLY $60 per day which of my trips could I repeat?"],
        "constraints": [],
        "expected_keys": ["Bali", "India", "60"],
        "context_hint": "Bali ($45) and India ($55) fit. Singapore ($130) and Dubai ($180) do not."
    },
    {
        "id": "M4",
        "family": "multi_hop",
        "turns": ["Which destination had the best combination of culture and affordability?"],
        "constraints": [],
        "expected_keys": ["India", "Bali"],
        "context_hint": "India ($55/day) has forts, temples, cooking classes, spice markets. Bali ($45/day) has temples and cooking."
    },
    {
        "id": "M5",
        "family": "multi_hop",
        "turns": ["Across all my trips where did I have the most wildlife encounters?"],
        "constraints": [],
        "expected_keys": ["Bali", "turtle"],
        "context_hint": "Bali Nusa Penida: sea turtles at Crystal Bay, dolphins at Agonda equivalent. Singapore: butterfly garden."
    },

    # ── Conversational (V1-V6) ────────────────────────────────────────────────
    # Constraint stated in turn 1 — must be honoured in turns 2 and 3
    # V0 has no memory so should fail CSR on turns 2 and 3
    # VA and VB have AgentState memory so should pass
    {
        "id": "V1",
        "family": "conversational",
        "turns": [
            "I hate crowded tourist spots",
            "Plan me 3 days in Bali",
            "What food should I try on that trip?"
        ],
        "constraints": ["avoid crowded places"],
        "expected_keys": ["Ubud", "warung", "local"],
        "context_hint": "Turns 2+3 must avoid Kuta, Seminyak. Should suggest Ubud, local warungs."
    },
    {
        "id": "V2",
        "family": "conversational",
        "turns": [
            "I have a budget of $50 per day",
            "Where should I eat in Singapore?",
            "Which of those options is cheapest?"
        ],
        "constraints": ["$50 per day budget"],
        "expected_keys": ["hawker", "maxwell"],
        "context_hint": "Must recommend hawker centres ($3-6 SGD). Not restaurants ($20+ SGD)."
    },
    {
        "id": "V3",
        "family": "conversational",
        "turns": [
            "I am vegetarian",
            "What can I eat in India based on my trip?",
            "Which city had the best vegetarian food?"
        ],
        "constraints": ["vegetarian"],
        "expected_keys": ["veg", "dal", "thali"],
        "context_hint": "Must only suggest vegetarian options. Dal, thali, aloo gobi all fine."
    },
    {
        "id": "V4",
        "family": "conversational",
        "turns": [
            "I cannot do physically demanding activities or long hikes",
            "What should I do in Bali?",
            "What about eating — where do you recommend?"
        ],
        "constraints": ["no strenuous hiking or physical activities"],
        "expected_keys": ["Ubud", "temple", "warung"],
        "context_hint": "Must avoid Kelingking Beach hike. Should suggest Ubud temples, cooking class, rice fields walk."
    },
    {
        "id": "V5",
        "family": "conversational",
        "turns": [
            "I strongly prefer authentic local experiences over touristy places",
            "Recommend me a city to visit in India",
            "What should I eat there?"
        ],
        "constraints": ["authentic local experiences not tourist traps"],
        "expected_keys": ["Jodhpur", "local"],
        "context_hint": "Should recommend Jodhpur or non-touristy parts of India. Avoid Red Fort queues, tourist restaurants."
    },
    {
        "id": "V6",
        "family": "conversational",
        "turns": [
            "I have a tight budget of $40 per day maximum",
            "Which of my trips would you suggest I repeat?",
            "What activities can I do there that fit that budget?"
        ],
        "constraints": ["$40 per day maximum budget"],
        "expected_keys": ["Bali", "warung"],
        "context_hint": "Only Bali ($45/day) comes close — and many days were under $40. India ($55) slightly over. Must stay budget-aware."
    },
]


# ── Metric 1: CSR (LLM-as-judge) ─────────────────────────────────────────────

def score_csr(response: str, constraints: list) -> dict:
    """
    Ask LLM whether each constraint was honoured in the response.
    Binary per constraint: 1 = honoured, 0 = violated or ignored.
    Consider it honoured even if not explicitly mentioned — as long
    as the response does not violate it.

    response: the final answer from the agent to be scored
    constraints: list of constraints the agent should have honoured

    Returns a dict with:
    - scores: {constraint: 1 or 0}
    - csr: average score across constraints (0 to 1)
    """
    # If no constraints, CSR is not applicable (None)
    if not constraints:
        return {"scores": {}, "csr": None}

    # Ask LLM to evaluate each constraint against the response
    scores = {}
    for constraint in constraints:
        prompt = f"""You are a strict evaluator.

Return ONLY valid JSON:
{{
  "satisfied": true or false
}}

Constraint: {constraint}

Response:
{response}"""

        raw = llm.invoke(prompt).strip()

        try:
            result    = json.loads(raw)
            satisfied = bool(result.get("satisfied", False))
        except Exception:
            # Fallback to keyword check if JSON parsing fails
            satisfied = "yes" in raw.lower() or "true" in raw.lower()

        scores[constraint] = 1 if satisfied else 0

    csr = sum(scores.values()) / len(scores)
    return {"scores": scores, "csr": round(csr, 2)}


# Metric 2: Relevance (LLM-as-judge) 
def score_relevance(query: str, response: str) -> int:
    """
    Ask LLM to score how well the response answers the query based on personal travel details.
    Scores 1-5:
    1 = completely generic, could apply to anyone
    2 = slightly personalised but mostly generic
    3 = some personal details used
    4 = mostly personal and specific
    5 = highly specific, uses real personal details throughout

    query: the original user query
    response: the final answer from the agent to be scored

    Returns an integer relevance score from 1 to 5.
    """
    prompt = f"""Score this travel assistant response from 1 to 5.

Criteria: How well does it use specific personal details from the user's
own travel history (real place names, real costs, real experiences)
rather than giving generic travel advice?

1 = completely generic, could apply to anyone
2 = slightly personalised but mostly generic
3 = some personal details used
4 = mostly personal and specific
5 = highly specific, uses real personal details throughout

Question: "{query}"
Response: "{response}"

Reply with ONLY a single number 1-5:"""

    raw   = llm.invoke(prompt).strip()
    match = re.search(r"[1-5]", raw)
    return int(match.group()) if match else 3


# Metric 3: Keyword Match
def score_keywords(response: str, expected_keys: list) -> dict:
    """
    Check how many of the expected keywords are present in the response.

    response: the final answer from the agent to be scored
    expected_keys: list of keywords that should ideally be mentioned in the response

    Returns a dict with:
    - hits: list of expected keywords found in the response
    - rate: percentage of expected keywords found (0 to 1)

        Note: This is a simple string match and does not account for synonyms or context.
    """
    if not expected_keys:
        return {"hits": [], "rate": None}

    # Count how many expected keywords are present in the response (case-insensitive)
    hits = [kw for kw in expected_keys if kw.lower() in response.lower()]
    rate = round(len(hits) / len(expected_keys), 2)
    return {
        "hits": hits,
        "rate": rate
    }



def run_single_turn(state, app, query: str, turn_idx: int, constraints: list) -> tuple:
    """
    Runs a single turn of the conversation with the agent, including CSR scoring if applicable.

    state: the current AgentState
    app: the LangGraph agent application
    query: the user query for this turn
    turn_idx: the index of the current turn (0-based)
    constraints: list of constraints to be scored for CSR (only for conversational queries)

    returns the updated state and a dict with turn results including response, latency, and CSR score if applicable.
    """
    state["query"] = query
    state          = run_turn(state, app)
    last_response  = state["messages"][-1].content
    latency        = state["latency"]

    print(f"    Response : {last_response[:80]}...")
    print(f"    Latency  : {latency}s")

    # CSR scored on every turn after turn 1
    # Turn 1 just states the constraint — nothing to score yet
    # Turn 2+ must honour it — this is where V0 fails
    turn_csr = None
    if constraints and turn_idx > 0:
        csr_result = score_csr(last_response, constraints)
        turn_csr   = csr_result["csr"]
        print(f"    CSR turn {turn_idx + 1}: {turn_csr}")

    turn_result = {
        "turn":     turn_idx + 1,
        "query":    query,
        "response": last_response,
        "latency":  latency,
        "csr":      turn_csr
    }

    return state, turn_result

def run_test_case(test: dict, app, mode: str) -> list:
    """
    Runs a single test case through the agent, turn by turn, and collects results.
    
    test: dict containing test case details (id, family, turns, constraints, expected_keys)
    app: the LangGraph agent application
    mode: the variant being tested (plain, chunks, episodes)
     
    returns a list of turn results with response, latency, and CSR score if applicable.
    """
    print(f"\n  [{test['id']}] {test['family'].upper()}")

    state        = build_initial_state(mode)
    turn_results = []

    for turn_idx, query in enumerate(test["turns"]):
        print(f"    Turn {turn_idx + 1}: {query[:60]}")
        state, turn_result = run_single_turn(state, app, query, turn_idx, test["constraints"])
        turn_results.append(turn_result)

    return turn_results


def score_final_response(test: dict, turn_results: list) -> dict:
    """
    Scores the final response of the agent on CSR, relevance, keyword match, and latency.
    
    test: dict containing test case details (id, family, turns, constraints, expected_keys)
    turn_results: list of dicts with results from each turn, including the final response and latency
      
    returns a dict with final CSR, relevance score, keyword match rate, and average latency.
    """
    last_response = turn_results[-1]["response"]

    final_csr   = score_csr(last_response, test["constraints"])
    relevance   = score_relevance(test["turns"][-1], last_response)
    kw_result   = score_keywords(last_response, test["expected_keys"])
    avg_latency = round(sum(t["latency"] for t in turn_results) / len(turn_results), 2)

    scored_turns = [
        t["csr"] for t in turn_results
        if isinstance(t.get("csr"), (int, float))
    ]
    avg_turn_csr = round(sum(scored_turns) / len(scored_turns), 2) if scored_turns else None

    print(f"    Final CSR    : {final_csr['csr']}")
    print(f"    Avg turn CSR : {avg_turn_csr}")
    print(f"    Relevance    : {relevance}/5")
    print(f"    Keywords     : {len(kw_result['hits'])}/{len(test['expected_keys'])} ({kw_result['rate']})")
    print(f"    Avg latency  : {avg_latency}s")

    return {
        "final_csr":    final_csr["csr"],
        "avg_turn_csr": avg_turn_csr,
        "csr_detail":   final_csr["scores"],
        "relevance":    relevance,
        "keyword_rate": kw_result["rate"],
        "keyword_hits": kw_result["hits"],
        "avg_latency":  avg_latency,
    }

def run_variant(mode: str) -> list:
    """
    Runs all test cases for a given variant and collects results.
    
    mode: the variant being tested (plain, chunks, episodes)
    
    returns a list of result records for each test case, including scores and details.
    """
    print(f"\n{'='*55}")
    print(f"  RUNNING {VARIANT_LABELS.get(mode,'').upper()}")
    print(f"{'='*55}")

    app     = build_graph(mode)
    results = []

    for test in TEST_SET:
        turn_results = run_test_case(test, app, mode)
        scores = score_final_response(test, turn_results)
        results.append({
            "id": test["id"],
            "family": test["family"],
            "mode": mode,
            "turns": turn_results,
            **scores
        })
    return results


def compute_summary(results: list, mode: str) -> dict:
    """
    Computes average scores overall and by query family for a given variant.
    
    results: list of result records for each test case in this variant
    mode: the variant being summarized (plain, chunks, episodes)
    
    returns a dict with overall averages and averages by family.
    """
    families = ["factual", "cross_modal", "multi_hop", "conversational"]
    summary  = {"mode": mode, "overall": {}, "by_family": {}}

    all_relevance = [r["relevance"]    for r in results]
    all_latency   = [r["avg_latency"]  for r in results]
    all_keywords  = [r["keyword_rate"] for r in results if r["keyword_rate"] is not None]
    csr_list      = [r["final_csr"]    for r in results if r["final_csr"]    is not None]
    turn_csr_list = [r["avg_turn_csr"] for r in results if r["avg_turn_csr"] is not None]

    summary["overall"] = {
        "avg_relevance":    round(sum(all_relevance) / len(all_relevance), 2),
        "avg_latency":      round(sum(all_latency) / len(all_latency), 2),
        "avg_keyword_rate": round(sum(all_keywords) / len(all_keywords), 2) if all_keywords else "N/A",
        "avg_final_csr":    round(sum(csr_list) / len(csr_list), 2)          if csr_list      else "N/A",
        "avg_turn_csr":     round(sum(turn_csr_list) / len(turn_csr_list), 2) if turn_csr_list else "N/A",
        "total_queries":    len(results)
    }

    for family in families:
        fam          = [r for r in results if r["family"] == family]
        if not fam:
            continue
        fam_csr      = [r["final_csr"]    for r in fam if r["final_csr"]    is not None]
        fam_turn_csr = [r["avg_turn_csr"] for r in fam if r["avg_turn_csr"] is not None]
        fam_kw       = [r["keyword_rate"] for r in fam if r["keyword_rate"] is not None]

        summary["by_family"][family] = {
            "avg_relevance":    round(sum(r["relevance"]   for r in fam) / len(fam), 2),
            "avg_latency":      round(sum(r["avg_latency"] for r in fam) / len(fam), 2),
            "avg_keyword_rate": round(sum(fam_kw) / len(fam_kw), 2)             if fam_kw       else "N/A",
            "avg_final_csr":    round(sum(fam_csr) / len(fam_csr), 2)           if fam_csr      else "N/A",
            "avg_turn_csr":     round(sum(fam_turn_csr) / len(fam_turn_csr), 2) if fam_turn_csr else "N/A"
        }

    return summary


# Save Results 

def save_results(all_results: list, summaries: list):
    # Full JSON
    json_path = OUTPUT_DIR / "results.json"
    json_path.write_text(json.dumps(
        {"results": all_results, "summaries": summaries}, indent=2
    ))
    print(f"\n  ✓ Full results     → {json_path}")

    # Summary CSV
    csv_path = OUTPUT_DIR / "results_summary.csv"
    r0 = {r["id"]: r for r in all_results if r["mode"] == "plain"}
    ra = {r["id"]: r for r in all_results if r["mode"] == "chunks"}
    rb = {r["id"]: r for r in all_results if r["mode"] == "episodes"}

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ID", "Family",
            "V0 Relevance", "VA Relevance", "VB Relevance",
            "V0 Keywords",  "VA Keywords",  "VB Keywords",
            "V0 CSR",       "VA CSR",       "VB CSR",
            "V0 Latency",   "VA Latency",   "VB Latency",
        ])
        for test in TEST_SET:
            tid = test["id"]
            writer.writerow([
                tid, test["family"],
                r0.get(tid, {}).get("relevance",    ""),
                ra.get(tid, {}).get("relevance",    ""),
                rb.get(tid, {}).get("relevance",    ""),
                r0.get(tid, {}).get("keyword_rate", ""),
                ra.get(tid, {}).get("keyword_rate", ""),
                rb.get(tid, {}).get("keyword_rate", ""),
                r0.get(tid, {}).get("final_csr",    "N/A"),
                ra.get(tid, {}).get("final_csr",    "N/A"),
                rb.get(tid, {}).get("final_csr",    "N/A"),
                r0.get(tid, {}).get("avg_latency",  ""),
                ra.get(tid, {}).get("avg_latency",  ""),
                rb.get(tid, {}).get("avg_latency",  ""),
            ])
    print(f"  ✓ Summary CSV      → {csv_path}")

    # Per-turn CSR CSV — your headline finding
    csr_path = OUTPUT_DIR / "csr_per_turn.csv"
    conv_ids = [t["id"] for t in TEST_SET if t["family"] == "conversational"]

    with open(csr_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Test ID", "Turn", "Query", "V0 CSR", "VA CSR", "VB CSR"])

        for tid in conv_ids:
            v0_turns = next((r["turns"] for r in all_results if r["id"] == tid and r["mode"] == "plain"),    [])
            va_turns = next((r["turns"] for r in all_results if r["id"] == tid and r["mode"] == "chunks"),   [])
            vb_turns = next((r["turns"] for r in all_results if r["id"] == tid and r["mode"] == "episodes"), [])

            max_turns = max(len(v0_turns), len(va_turns), len(vb_turns), 1)
            for i in range(max_turns):
                v0t = v0_turns[i] if i < len(v0_turns) else {}
                vat = va_turns[i] if i < len(va_turns) else {}
                vbt = vb_turns[i] if i < len(vb_turns) else {}
                writer.writerow([
                    tid, i + 1,
                    v0t.get("query", "")[:50],
                    v0t.get("csr", "N/A"),
                    vat.get("csr", "N/A"),
                    vbt.get("csr", "N/A"),
                ])
    print(f"  ✓ Per-turn CSR CSV → {csr_path}")


#  Print Comparison Table

def print_comparison(summaries: list):
    v0 = next((s for s in summaries if s["mode"] == "plain"),    {}).get("overall", {})
    va = next((s for s in summaries if s["mode"] == "chunks"),   {}).get("overall", {})
    vb = next((s for s in summaries if s["mode"] == "episodes"), {}).get("overall", {})

    print(f"\n{'='*65}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Metric':<32} {'V0':>10} {'VA':>10} {'VB':>10}")
    print(f"  {'-'*62}")
    print(f"  {'Avg Relevance (1-5)':<32} {str(v0.get('avg_relevance','')):>10} {str(va.get('avg_relevance','')):>10} {str(vb.get('avg_relevance','')):>10}")
    print(f"  {'Avg Keyword Match':<32} {str(v0.get('avg_keyword_rate','')):>10} {str(va.get('avg_keyword_rate','')):>10} {str(vb.get('avg_keyword_rate','')):>10}")
    print(f"  {'Avg Final CSR':<32} {str(v0.get('avg_final_csr','N/A')):>10} {str(va.get('avg_final_csr','N/A')):>10} {str(vb.get('avg_final_csr','N/A')):>10}")
    print(f"  {'Avg Turn CSR':<32} {str(v0.get('avg_turn_csr','N/A')):>10} {str(va.get('avg_turn_csr','N/A')):>10} {str(vb.get('avg_turn_csr','N/A')):>10}")
    print(f"  {'Avg Latency (s)':<32} {str(v0.get('avg_latency','')):>10} {str(va.get('avg_latency','')):>10} {str(vb.get('avg_latency','')):>10}")

    print(f"\n  By Query Family")
    print(f"  {'-'*62}")
    for family in ["factual", "cross_modal", "multi_hop", "conversational"]:
        f0 = next((s for s in summaries if s["mode"] == "plain"),    {}).get("by_family", {}).get(family, {})
        fa = next((s for s in summaries if s["mode"] == "chunks"),   {}).get("by_family", {}).get(family, {})
        fb = next((s for s in summaries if s["mode"] == "episodes"), {}).get("by_family", {}).get(family, {})

        print(f"\n  {family.upper()}")
        print(f"    {'Relevance':<28} {str(f0.get('avg_relevance','')):>10} {str(fa.get('avg_relevance','')):>10} {str(fb.get('avg_relevance','')):>10}")
        print(f"    {'Keywords':<28} {str(f0.get('avg_keyword_rate','')):>10} {str(fa.get('avg_keyword_rate','')):>10} {str(fb.get('avg_keyword_rate','')):>10}")
        print(f"    {'Latency (s)':<28} {str(f0.get('avg_latency','')):>10} {str(fa.get('avg_latency','')):>10} {str(fb.get('avg_latency','')):>10}")
        if family == "conversational":
            print(f"    {'Final CSR':<28} {str(f0.get('avg_final_csr','N/A')):>10} {str(fa.get('avg_final_csr','N/A')):>10} {str(fb.get('avg_final_csr','N/A')):>10}")
            print(f"    {'Avg Turn CSR':<28} {str(f0.get('avg_turn_csr','N/A')):>10} {str(fa.get('avg_turn_csr','N/A')):>10} {str(fb.get('avg_turn_csr','N/A')):>10}")

    print(f"\n{'='*65}\n")


# Main 
def run_evaluation(modes: list):
    print("\n" + "="*65)
    print("  MEMORYLENS EVALUATION (LLM-as-judge)")
    print("="*65)
    print(f"  Variants    : {', '.join(modes)}")
    print(f"  Test cases  : {len(TEST_SET)} — "
          f"{len([t for t in TEST_SET if t['family']=='factual'])} factual, "
          f"{len([t for t in TEST_SET if t['family']=='cross_modal'])} cross-modal, "
          f"{len([t for t in TEST_SET if t['family']=='multi_hop'])} multi-hop, "
          f"{len([t for t in TEST_SET if t['family']=='conversational'])} conversational")
    print(f"  Metrics     : Relevance, Keywords, CSR (per-turn + final), Latency")
    print(f"  Scoring     : LLM-as-judge (llava)")
    print(f"  Memory      : AgentState only — resets per test case")

    all_results = []
    summaries   = []

    for mode in modes:
        results = run_variant(mode)
        summary = compute_summary(results, mode)
        all_results.extend(results)
        summaries.append(summary)

    save_results(all_results, summaries)
    print_comparison(summaries)
    print(f"  Results saved to evaluation/ folder.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MemoryLens Evaluation")
    parser.add_argument(
        "--mode",
        choices=["plain", "chunks", "episodes", "all"],
        default="all",
        help="Which variant to run (default: all)"
    )
    args  = parser.parse_args()
    modes = ["plain", "chunks", "episodes"] if args.mode == "all" else [args.mode]
    run_evaluation(modes=modes)