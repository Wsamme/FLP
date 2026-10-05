"""Rebuild instances/MANIFEST.md (SHA-256 of every instance and source file)."""
import glob, hashlib, json, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
L = ["# Instance manifest", "", "| File | SHA-256 | Status | Data source |", "|---|---|---|---|"]
for f in sorted(glob.glob("instances/uaflp/*.json")) + sorted(glob.glob("instances/sources/*/*")):
    h = hashlib.sha256(open(f, "rb").read()).hexdigest()
    if f.endswith(".json"):
        d = json.load(open(f))
        L.append(f"| `{f}` | `{h}` | {d['provenance']['status']} | {d['provenance']['data']} |")
    else:
        L.append(f"| `{f}` | `{h}` | source | Mendeley Data DOI 10.17632/bv3zywjyyj.1, CC BY 4.0 |")
L += ["", "Regenerate with `python scripts/manifest.py` after any change to instance files."]
open("instances/MANIFEST.md", "w").write("\n".join(L) + "\n")
