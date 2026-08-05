import sys
from rag.pipeline import ask

query = " ".join(sys.argv[1:]) or input("Question: ")
result = ask(query)

print("\nAnswer:", result["answer"])
print("\nSources:")
for s in result["sources"]:
    print("-", s.get("source", "unknown"))