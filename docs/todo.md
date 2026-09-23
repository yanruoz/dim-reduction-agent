## Progress
- [x] Repo + env set up
- [x] loaders.py: pbmc3k, pathmnist loading to common (X, y, metadata) interface
- [x] profiler.py: generic profile_dataset(X, y, metadata), handles sparse + dense
- [x] API key working, test_api.py confirms connectivity
- [ ] NEXT: build the planner — one LLM call that takes a profile dict and
      returns a JSON plan (preprocessing steps + methods + hyperparameters)
- [ ] Define tool schemas (run_reduction, evaluate_embedding, etc.)
- [ ] Build the agent loop (tool_use round-trip)
- [ ] Report generation