import socket, sys, re
sys.path.insert(0, "tests"); sys.path.insert(0, ".")
from herding import explore as ex
def blocked(*a, **k): raise OSError("blocked")
socket.socket.connect = blocked
def nod(t, s): raise OSError("no")
def noi(t, p, i): raise OSError("no")
ex._yahoo = nod; ex._yahoo_intraday = noi
from streamlit.testing.v1 import AppTest
def text(at):
    out = []
    for kind in ("title","header","subheader","markdown","caption","info","warning","success","error","text"):
        for e in getattr(at, kind):
            out.append(f"[{kind}] {e.value}")
    for e in at.exception: out.append(f"[EXC] {e.value}")
    return "\n".join(out)
pages = sys.argv[1:] or ["web/page_guide.py","web/page_story.py","web/page_checks.py","web/page_sources.py","web/page_explore.py"]
for p in pages:
    at = AppTest.from_file("/Users/avishjain/02-120 Project/app.py", default_timeout=900); at.run(); at.switch_page(p).run()
    name = p.split("/")[-1].replace(".py","")
    open(f"testing_checks/review_b/{name}.txt","w").write(text(at))
    print(p, "exceptions:", [e.value for e in at.exception], "chars", len(text(at)))
# old link with party key
at = AppTest.from_file("/Users/avishjain/02-120 Project/app.py", default_timeout=900)
for k,v in {"market":"S&P 500","a1":"0.8","party":"3.5","seed":"1"}.items(): at.query_params[k]=v
at.run(); at.switch_page("web/page_explore.py").run()
print("party link exceptions:", [e.value for e in at.exception])
