#!/usr/bin/env python3
"""Inspect each reasoning stage ("agent") of a bioavailability_ma run.

A run has three LLM stages, all the same model:
  1. single-molecule  -> single_molecule_reasoning_output.json  (physicochemical prior)
  2. group branches    -> group_reasoning_outputs.jsonl          (one call per endpoint group)
  3. final synthesis   -> final_reasoning_output.json            (integrates 1 + 2)

The final stage's *_assessment fields are written by the final call, each drawing on the
matching group branch (transporter_efflux_assessment <- Fg, etc.). Every stage stores its
private chain-of-thought in `reasoning_content` and its answer JSON in `content`.

Usage:
  python view_run.py <run_dir>                 # tree overview of all stages
  python view_run.py <run_dir> single          # single-molecule stage in full
  python view_run.py <run_dir> final           # final synthesis stage in full
  python view_run.py <run_dir> group <substr>  # one group branch (e.g. Fg, efflux, direct)
  python view_run.py <run_dir> map             # each final *_assessment next to its source group
  python view_run.py <run_dir> prompt group <substr>   # exact system+user prompt sent for a group
  python view_run.py <run_dir> prompt single|final     # exact prompt sent for those stages

A <run_dir> is any .../runs/..._idxNNNNN directory under
outputs/paper/molecular_evidence_agent/<batch>/.../runs/.
"""
import json
import sys

# Which final *_assessment field is primarily fed by which group_id substring.
ASSESSMENT_TO_GROUP = {
    "absorption_and_permeability_assessment": "Fa.",
    "solubility_and_dissolution_assessment": "Fa.",
    "metabolism_first_pass_and_clearance_assessment": "Fh.",
    "transporter_efflux_assessment": "Fg.",
    "direct_oral_bioavailability_analog_assessment": "Observed.direct",
}


def load_groups(run):
    return [json.loads(l) for l in open(f"{run}/group_reasoning_outputs.jsonl") if l.strip()]


def llm_of(o):
    return o.get("llm") or o


def hr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def dump_stage(name, o, show_tools=False):
    llm = llm_of(o)
    hr(f"STAGE: {name}   (status={o.get('status', '?')}, model={llm.get('model', '?')})")
    rc = llm.get("reasoning_content", "")
    print("\n--- PRIVATE REASONING (chain-of-thought) ---")
    print(rc if rc else "(none captured)")
    if show_tools:
        for i, tr in enumerate(llm.get("tool_results") or [], 1):
            print(f"\n--- TOOL RESULT {i}: {tr.get('tool_name')} (status={tr.get('status')}) ---")
            print(tr.get("content"))
    print("\n--- STRUCTURED OUTPUT (JSON the agent returned) ---")
    print(json.dumps(llm.get("content"), indent=2, ensure_ascii=False))


def find_group(groups, key):
    return next(g for g in groups if key.lower() in g["group_id"].lower())


def dump_prompt(name, o):
    """Print the exact system + user messages the model received for a stage.

    The user message is the literal string sent. For the legacy JSON format a
    pretty-printed copy follows; the new text formats (morganfingerprint,
    assay_transfer_tool) are already human-readable and printed as-is.
    """
    llm = llm_of(o)
    msgs = llm["messages"]
    system = next(m for m in msgs if m.get("role") == "system")
    user = next(m for m in msgs if m.get("role") == "user")
    hr(f"PROMPT: {name}")
    print("\n--- SYSTEM MESSAGE (verbatim) ---")
    print(system["content"])
    try:
        parsed = json.loads(user["content"])
    except (ValueError, TypeError):
        parsed = None
    if parsed is None:
        print("\n--- USER MESSAGE (verbatim text as sent) ---")
        print(user["content"])
    else:
        print("\n--- USER MESSAGE (verbatim string as sent -- compact JSON) ---")
        print(user["content"])
        print("\n--- USER MESSAGE (same content, pretty-printed) ---")
        print(json.dumps(parsed, indent=2, ensure_ascii=False))


def tree(run):
    sm = json.load(open(f"{run}/single_molecule_reasoning_output.json"))
    fin = json.load(open(f"{run}/final_reasoning_output.json"))
    groups = load_groups(run)
    hr("REASONING STAGES ('agents') IN THIS RUN")

    def line(label, o, extra=""):
        llm = llm_of(o)
        rc = len(llm.get("reasoning_content") or "")
        ntc = len(llm.get("tool_calls") or [])
        print(f"  {label:52s} reasoning={rc:>6}ch  tool_calls={ntc}  {extra}")

    print("\n1) SINGLE-MOLECULE (physicochemical prior, 1 call)")
    line("single_molecule", sm, f"conf={llm_of(sm)['content'].get('confidence')}")
    print("\n2) GROUP BRANCHES (one call per endpoint group -- the closest thing to 'subagents')")
    for g in groups:
        c = llm_of(g)["content"]
        line(g["group_id"], g, f"transfer={c.get('transferability')} dir={c.get('evidence_direction')}")
    print("\n3) FINAL SYNTHESIS (integrates all of the above, 1 call)")
    fc = llm_of(fin)["content"]
    line("final", fin, f"pred={fc.get('bioavailability_prediction')} conf={fc.get('confidence')}")
    print("\nDrill in with:  view_run.py <run_dir> group Fg   (or: single | final | map)")


def mapping(run):
    fin = json.load(open(f"{run}/final_reasoning_output.json"))
    groups = load_groups(run)
    fc = llm_of(fin)["content"]
    hr("FINAL *_assessment  <-  SOURCE GROUP BRANCH")
    for field, gkey in ASSESSMENT_TO_GROUP.items():
        print(f"\n### {field}")
        try:
            g = find_group(groups, gkey)
            gc = llm_of(g)["content"]
            print(f"  source group : {g['group_id']}")
            print(f"  group verdict: transferability={gc.get('transferability')} "
                  f"direction={gc.get('evidence_direction')} confidence={gc.get('confidence')}")
        except StopIteration:
            print(f"  source group : (no group matching '{gkey}')")
        print(f"  final says   : {fc.get(field, '(missing)')}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    run = sys.argv[1]
    what = sys.argv[2] if len(sys.argv) > 2 else "tree"
    if what == "tree":
        tree(run)
    elif what == "single":
        dump_stage("single_molecule", json.load(open(f"{run}/single_molecule_reasoning_output.json")))
    elif what == "final":
        dump_stage("final", json.load(open(f"{run}/final_reasoning_output.json")))
    elif what == "group":
        dump_stage(find_group(load_groups(run), sys.argv[3])["group_id"],
                   find_group(load_groups(run), sys.argv[3]), show_tools=True)
    elif what == "map":
        mapping(run)
    elif what == "prompt":
        stage = sys.argv[3] if len(sys.argv) > 3 else "group"
        if stage == "single":
            dump_prompt("single_molecule", json.load(open(f"{run}/single_molecule_reasoning_output.json")))
        elif stage == "final":
            dump_prompt("final", json.load(open(f"{run}/final_reasoning_output.json")))
        else:  # group <substr>
            key = sys.argv[4] if len(sys.argv) > 4 else stage
            dump_prompt(find_group(load_groups(run), key)["group_id"],
                        find_group(load_groups(run), key))
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
