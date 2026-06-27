"""Unpack the runner-authoring workflow result into baselines_ext/<method>/{Dockerfile,run.py,NOTES.md}."""
import json, os, re, sys

src = json.load(open(sys.argv[1]))
runners = src.get("result", src).get("runners", [])


def tag(m):
    ml = m.lower()
    for key, t in [("poco", "poco"), ("nksr", "nksr"), ("neural kernel", "nksr"),
                   ("shape as points", "sap"), ("shape-as-points", "sap"),
                   ("neural-pull", "neuralpull"), ("neuralpull", "neuralpull"), ("neural pull", "neuralpull"),
                   ("convolutional occupancy", "convonet"), ("convonet", "convonet"),
                   ("cap-udf", "capudf"), ("capudf", "capudf"), ("cap udf", "capudf")]:
        if key in ml:
            return t
    if ml.startswith("sap"):
        return "sap"
    return re.sub(r"\W+", "_", ml)[:12]


rows = []
for r in runners:
    t = tag(r.get("method", "?")); d = f"baselines_ext/{t}"; os.makedirs(d, exist_ok=True)
    if r.get("dockerfile"): open(f"{d}/Dockerfile", "w", newline="\n").write(r["dockerfile"])
    if r.get("run_py"): open(f"{d}/run.py", "w", newline="\n").write(r["run_py"])
    notes = "".join(f"## {k}\n\n{r.get(k)}\n\n" for k in
                    ("method", "feasibility", "blocker", "build_cmd", "run_cmd", "weights_note", "ood_note", "caveats")
                    if r.get(k))
    open(f"{d}/NOTES.md", "w", newline="\n").write(notes)
    rows.append((t, r.get("feasibility", "?"), r.get("build_cmd", "").strip().splitlines()[0] if r.get("build_cmd") else ""))
    print(f"{t:11s} {r.get('feasibility','?')}", flush=True)

open("baselines_ext/RUNNERS.md", "w", newline="\n").write(
    "# Dockerised learned baselines\n\n" +
    "\n".join(f"- **{t}** — {f}  (`baselines_ext/{t}/`)" for t, f, _ in rows) + "\n")
print(f"\ninstalled {len(rows)} runners -> baselines_ext/", flush=True)
