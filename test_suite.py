"""
test_suite.py — Unit tests and validation for Vera bot.py, conversation_handlers.py, and test pairs.
"""

import json
from pathlib import Path
import bot
import conversation_handlers as ch

def test_all_pairs():
    print("--- Testing all 30 canonical test pairs ---")
    pairs_file = Path("dataset/expanded/test_pairs.json")
    if not pairs_file.exists():
        print("Error: test_pairs.json not found")
        return False
        
    pairs = json.load(open(pairs_file))["pairs"]
    print(f"Loaded {len(pairs)} test pairs.")
    
    categories = {}
    for f in Path("dataset/categories").glob("*.json"):
        cat = json.load(open(f))
        categories[cat["slug"]] = cat

    submission_rows = []
    
    for p in pairs:
        tid = p["test_id"]
        m_id = p["merchant_id"]
        trg_id = p["trigger_id"]
        c_id = p.get("customer_id")
        
        m_file = Path(f"dataset/expanded/merchants/{m_id}.json")
        t_file = Path(f"dataset/expanded/triggers/{trg_id}.json")
        c_file = Path(f"dataset/expanded/customers/{c_id}.json") if c_id else None
        
        merchant = json.load(open(m_file))
        trigger = json.load(open(t_file))
        category = categories[merchant["category_slug"]]
        customer = json.load(open(c_file)) if c_file and c_file.exists() else None
        
        result = bot.compose(category, merchant, trigger, customer)
        
        assert result is not None, f"compose returned None for {tid}"
        assert "body" in result and result["body"], f"Missing body in {tid}"
        assert "cta" in result, f"Missing cta in {tid}"
        assert "send_as" in result, f"Missing send_as in {tid}"
        assert "suppression_key" in result, f"Missing suppression_key in {tid}"
        assert "rationale" in result, f"Missing rationale in {tid}"
        
        # Verify taboo check
        taboos = category.get("voice", {}).get("vocab_taboo", [])
        body_lower = result["body"].lower()
        for taboo in taboos:
            clean_taboo = taboo.split("(")[0].strip().lower()
            if len(clean_taboo) > 4:
                assert clean_taboo not in body_lower, f"Taboo word '{clean_taboo}' found in {tid}: {result['body']}"
                
        print(f"[{tid}] PASS | Kind: {trigger['kind'][:20]:20} | Scope: {trigger['scope']:8} | send_as: {result['send_as']:18} | len: {len(result['body'])}")
        
        submission_rows.append({
            "test_id": tid,
            "body": result["body"],
            "cta": result["cta"],
            "send_as": result["send_as"],
            "suppression_key": result["suppression_key"],
            "rationale": result["rationale"],
        })

    # Save submission.jsonl
    with open("submission.jsonl", "w", encoding="utf-8") as f:
        for row in submission_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            
    print(f"\nSuccessfully wrote {len(submission_rows)} rows to submission.jsonl")
    return True

def test_conversation_handling():
    print("\n--- Testing multi-turn conversation states ---")
    state = ch.ConversationState(conversation_id="conv_test_1", owner_name="Dr. Meera", language_pref="en")
    
    # 1. Intent transition test
    res = ch.respond(state, "Ok lets do it. Whats next?")
    assert res["action"] == "send"
    actioning = ["done", "sending", "draft", "here", "confirm", "proceed", "next"]
    qualifying = ["would you", "do you", "can you tell", "what if", "how about"]
    body_low = res["body"].lower()
    assert any(w in body_low for w in actioning), f"No actioning word in {res['body']}"
    assert not any(w in body_low for w in qualifying), f"Qualifying word found in {res['body']}"
    print("[Intent Transition] PASS")
    
    # 2. Hostile handling test
    res2 = ch.respond(state, "Stop messaging me. This is useless spam.")
    assert res2["action"] == "end"
    print("[Hostility Handling] PASS")
    
    # 3. Auto-reply test
    state_auto = ch.ConversationState(conversation_id="conv_auto")
    r1 = ch.respond(state_auto, "Thank you for contacting us! Our team will respond shortly.")
    assert r1["action"] == "wait"
    assert r1.get("wait_seconds", 0) > 0
    print("[Auto-reply Turn 1: WAIT] PASS")
    
    r2 = ch.respond(state_auto, "Thank you for contacting us! Our team will respond shortly.")
    assert r2["action"] == "end"
    print("[Auto-reply Turn 2: END] PASS")
    
    # 4. Out-of-scope redirection test
    state_gst = ch.ConversationState(conversation_id="conv_gst")
    r_gst = ch.respond(state_gst, "Can you also file my GST return?")
    assert r_gst["action"] == "send"
    assert "ca" in r_gst["body"].lower() or "tax" in r_gst["body"].lower()
    print("[Out-of-scope Redirection] PASS")
    
    return True

if __name__ == "__main__":
    t1 = test_all_pairs()
    t2 = test_conversation_handling()
    if t1 and t2:
        print("\nALL UNIT TESTS PASSED PERFECTLY!")
