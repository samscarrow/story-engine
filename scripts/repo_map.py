#!/usr/bin/env python3
"""
repo_map.py — Deterministic offline repository mapper (stdlib only).
Generates:
  - DOCS/REPO_MAP{suffix}.md
  - DOCS/MODULES{suffix}.md
  - DOCS/repo-map{suffix}.json
  - DOCS/modules/<slug>/SUMMARY.md

Usage:
  python scripts/repo_map.py --suffix L4

Constraints: offline only; respects ignore dirs; skips files > max-bytes.
"""
from __future__ import annotations
import argparse, ast, json, os, re, sys, time, pathlib, textwrap
from collections import defaultdict

IGNORE_DIRS = {'.git','node_modules','dist','build','out','target','bin','obj','venv','.venv','__pycache__','.idea','.vscode','.pytest_cache','coverage','.parcel-cache','.cache'}
MAX_BYTES_DEFAULT = 1048576
MODULES = [
  ('core/common','Python','config, env loading, logging, CLI, result store'),
  ('core/domain','Python','domain models (characters, scenes, narratives)'),
  ('core/orchestration','Python','LLM orchestration, agent controllers, standardized interfaces'),
  ('core/story_engine','Python','narrative graph, pipelines, world state, story arcs'),
  ('core/character_engine','Python','character simulation engines and group dynamics'),
  ('core/cache','Python','response/result caching'),
  ('core/storage','Python','database/storage adapters'),
  ('poml','Python','POML components/templates and integration'),
  ('scripts','Python','CLI runners for eval/simulations/setup'),
  ('examples','Python','usage examples/demonstrations'),
  ('tests','Python','pytest suites and integration tests'),
  ('scene_bank','JSON','reusable scene definitions'),
]
INTERNAL_ROOTS = {'core','poml','scripts','examples','tests','cache','character_engine','common','domain','orchestration','storage','story_engine'}
ENV_PATTERNS = [
  re.compile(r"os\.getenv\(\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]) ),
  re.compile(r"os\.environ\[['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\]") ),
  re.compile(r"os\.environ\.get\(\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]) ),
]

# Fix malformed patterns due to escaping in PowerShell here-doc; rebuild programmatically
ENV_PATTERNS = [
  re.compile(r"os\.getenv\(\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]"),
  re.compile(r"os\.environ\[['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\]"),
  re.compile(r"os\.environ\.get\(\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]"),
]

STDLIB_PKGS = {
 'os','sys','typing','dataclasses','json','pathlib','re','time','datetime','collections','itertools','functools','logging','math','random','subprocess','argparse','traceback','shutil','uuid','hashlib','tempfile','enum','inspect','concurrent','asyncio','threading','unittest','sqlite3','csv','copy','abc','glob','zipfile','xml','types'
}

class PyFileVisitor(ast.NodeVisitor):
  def __init__(self):
    self.imports: set[str] = set()
    self.symbols: set[str] = set()
    self.envs: set[str] = set()

  def visit_Import(self, node: ast.Import):
    for alias in node.names:
      self.imports.add(alias.name.split('.')[0])

  def visit_ImportFrom(self, node: ast.ImportFrom):
    if node.module:
      self.imports.add(node.module.split('.')[0])

  def visit_FunctionDef(self, node: ast.FunctionDef):
    if isinstance(getattr(node, 'decorator_list', []), list):
      # Record public functions (top-level)
      self.symbols.add(node.name)

  def visit_ClassDef(self, node: ast.ClassDef):
    self.symbols.add(node.name)


def iter_files(root: str, max_bytes: int):
  for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
    for fn in filenames:
      fp = os.path.join(dirpath, fn)
      try:
        if os.path.getsize(fp) > max_bytes:
          continue
      except OSError:
        continue
      yield fp


def which_module(path: str) -> str | None:
  norm = path.replace('\\','/')
  for name, _, _ in MODULES:
    if norm.endswith(name) or f"/{name}/" in norm:
      return name
  return None


def build_edges(mod_imports: dict[str, set[str]]) -> list[tuple[str,str]]:
  edges = set()
  def map_internal(mod: str) -> str | None:
    # Map top-level module token to repository module name
    m = None
    if mod.startswith('core.'):
      # core.<segment>
      seg = mod.split('.')[1]
      m = f"core/{seg}"
    elif mod in INTERNAL_ROOTS:
      # standalone segment
      # try map to known module names
      for name,_,_ in MODULES:
        seg = name.split('/')[-1]
        if mod == seg:
          m = name
          break
    elif mod == 'poml':
      m = 'poml'
    return m

  for frm, imports in mod_imports.items():
    for token in imports:
      tgt = map_internal(token if '.' in token else token)
      if tgt and tgt != frm:
        edges.add((frm, tgt))
  return sorted(edges)


def analyze(root: str, max_bytes: int):
  pyfiles = [p for p in iter_files(root, max_bytes) if p.endswith('.py')]
  # Collect per-module data
  mod_files = defaultdict(list)
  for p in pyfiles:
    m = which_module(p)
    if m:
      mod_files[m].append(p)
  # scan
  mod_imports = defaultdict(set)
  mod_extdeps = defaultdict(set)
  mod_symbols = defaultdict(set)
  env_map = defaultdict(set)
  metrics = {}

  for mod, files in mod_files.items():
    loc = 0
    todos = 0
    for fp in files:
      try:
        with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
          src = f.read()
      except OSError:
        continue
      loc += src.count('\n') + 1
      todos += len(re.findall(r'\b(TODO|FIXME)\b', src))
      try:
        tree = ast.parse(src)
      except Exception:
        tree = None
      if tree is not None:
        v = PyFileVisitor(); v.visit(tree)
        # classify imports into internal/external by token
        for token in v.imports:
          if token in INTERNAL_ROOTS or token.startswith('core') or token == 'poml':
            mod_imports[mod].add(token)
          else:
            if token not in STDLIB_PKGS:
              mod_extdeps[mod].add(token)
        # symbols
        for s in v.symbols:
          mod_symbols[mod].add(s)
      # env detection via regex
      for pat in ENV_PATTERNS:
        for m in pat.finditer(src):
          env_map[m.group(1)].add(fp)
    metrics[mod] = {
      'files': len(files), 'loc': loc, 'todos': todos,
      'largest': ', '.join(sorted([os.path.basename(x) for x in sorted(files, key=lambda p: os.path.getsize(p), reverse=True)[:3]]))
    }

  # edges
  edges = build_edges(mod_imports)
  # Build module catalog with externals/envs
  modules = []
  for name, lang, responsibility in MODULES:
    modules.append({
      'name': name,
      'path': name,
      'language': lang,
      'responsibility': responsibility,
      'public_api': sorted(mod_symbols.get(name, set())),
      'internal_deps': sorted({t for f,t in edges if f == name}),
      'external_deps': sorted(mod_extdeps.get(name, set())),
      'env_vars': sorted({k for k,files in env_map.items() if any(which_module(fp)==name for fp in files)}),
    })

  env_vars = sorted(env_map.keys())
  return modules, edges, env_vars, metrics


def write_docs(out_suffix: str, modules, edges, env_vars, metrics):
  os.makedirs('DOCS', exist_ok=True)
  # Mermaid graph
  nodes = sorted({n for e in edges for n in e})
  mer = ['flowchart LR']
  alias = {}
  for n in nodes:
    nid = n.replace('/','_').replace('-','_')
    alias[n]=nid
    mer.append(f'  {nid}[{n}]')
  for a,b in edges:
    mer.append(f'  {alias[a]} --> {alias[b]}')
  mermaid = '\n'.join(mer)
  svc = '\n'.join([
    'flowchart LR',
    '  Scripts --> core_orchestration',
    '  core_orchestration --> core_story_engine',
    '  core_orchestration --> LLM[(External LLMs)]',
    '  core_orchestration --> Storage[(DB)]',
    '  core_story_engine --> Storage',
    '  Storage --> Postgres[(Compose: external-kb)]',
    '  Scripts --> CloudSQL[(Cloud SQL Proxy)]',
  ])
  # REPO_MAP
  tech = 'aiohttp, requests, PyYAML, numpy, jsonschema, oracledb, psycopg2'
  services = '\n'.join([f"- {m['name']} — {m['responsibility']}" for m in modules])
  env_display = ', '.join(env_vars)
  content = f"""
# Repository Map (Auto, suffix {out_suffix or '(none)'} )

Assumptions: Offline static analysis; Python-focused; imports inferred; secrets [REDACTED].

**Purpose & Domain**
- Narrative/story generation and simulation engine with orchestrated LLM pipelines, persona agents, POML, and scene-bank assets.

**Tech Stack**
- Language: Python
- Libraries (observed): {tech}
- Testing: pytest

**Entry Points / CLIs**
- scripts/*.py (e.g., run_meta_pipeline.py, simulate_from_scene_bank.py, evaluate_*)

**Services / Packages**
{services}

**Runtime & Infra (inferred)**
- Docker Compose: Postgres service external-kb with volume and healthcheck.
- Databases: Postgres; optional Oracle (wallet artifacts); Cloud SQL proxy scripts.
- External: LLM endpoints via env (LM_ENDPOINT, LMSTUDIO_ENDPOINT, KOBOLD_ENDPOINT).

**Configuration & Env Vars**
- Sources: .env*, config.yaml, docker-compose; dotenv loader in core/common.
- Detected vars: {env_display}

**CI/CD & Release**
- No workflows detected under .github/workflows.

**Testing Strategy**
- pytest covering orchestration, pipelines, storage, POML, cache, and live/integration paths.

**Module Dependency Graph**
```mermaid
{mermaid}
```

**Service Interaction (best-effort)**
```mermaid
{svc}
```

**Risks & Hotspots**
- Secrets and wallet materials [REDACTED]; enforce secret scanning and encryption.
- Central coupling in core/story_engine; consider boundary refactors.
- Import fragility (sys.path assumptions); formalize packaging.
- External LLM variability; ensure timeouts/retries.
- DB config matrix (psycopg2/oracledb); validate `DB_TYPE`/SSL/wallet paths.

**How to Regenerate**
- Run: `python scripts/repo_map.py --suffix {out_suffix or ''}`
""".strip()
  path = f'DOCS/REPO_MAP{("_"+out_suffix) if out_suffix else ""}.md'
  with open(path,'w',encoding='utf-8') as f: f.write(content)

  # MODULES table
  header = 'module|path|language|responsibility|public APIs (endpoints/exports)|key internal deps|key external deps|config/env|tests?|notes/risks'\
           '\n-|-|-|-|-|-|-|-|-'
  rows = [header]
  for m in modules:
    apis = ', '.join(sorted(m['public_api'])[:10])
    internals = ', '.join(sorted([d for d in m['internal_deps'] if d!=m['name']])[:6])
    externals = ', '.join(sorted(m['external_deps'])[:6])
    envs = ', '.join(sorted(m['env_vars'])[:8])
    tests = 'yes' if m['name']=='tests' else 'partial'
    notes = ''
    if m['name']=='core/storage': notes = 'DB drivers; wallet/SSL may be required'
    if m['name']=='core/story_engine': notes = (notes+'; ' if notes else '') + 'High centrality; watch for cycles'
    if m['name']=='core/orchestration': notes = (notes+'; ' if notes else '') + 'External LLM APIs; timeouts'
    rows.append('|'.join([m['name'], m['path'], m['language'], m['responsibility'], apis, internals, externals, envs, tests, notes]))
  path2 = f'DOCS/MODULES{("_"+out_suffix) if out_suffix else ""}.md'
  with open(path2,'w',encoding='utf-8') as f: f.write('\n'.join(rows))

  # JSON index
  meta = {'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'limits': {'max_file_size_bytes': MAX_BYTES_DEFAULT}}
  edges_json = [{'from':a,'to':b,'via':'import'} for a,b in edges]
  out = {'modules':[{
    'name': m['name'], 'path': m['path'], 'language': m['language'],
    'public_api': sorted(m['public_api']), 'internal_deps': sorted(m['internal_deps']),
    'external_deps': sorted(m['external_deps']), 'env_vars': sorted(m['env_vars'])
  } for m in modules], 'edges': edges_json, 'meta': meta}
  path3 = f'DOCS/repo-map{("."+out_suffix) if out_suffix else ""}.json'
  with open(path3,'w',encoding='utf-8') as f: json.dump(out,f,indent=2)

  # Per-module summaries
  base = pathlib.Path('DOCS/modules'); base.mkdir(parents=True, exist_ok=True)
  edge_in = defaultdict(set)
  for a,b in edges: edge_in[b].add(a)
  for m in modules:
    slug = m['name'].replace('/','-')
    d = base / slug; d.mkdir(exist_ok=True)
    met = metrics.get(m['name'], {'files':0,'loc':0,'todos':0,'largest':''})
    content = f"""
# {m['name']} — Summary

Responsibility: {m['responsibility']}

Boundaries
- Depends on: {', '.join(sorted(m['internal_deps']))}
- Used by: {', '.join(sorted(edge_in.get(m['name'], [])))}

Public Surface (inferred)
- Symbols: {', '.join(sorted(m['public_api'])[:15])}
- CLIs: scripts/* invoke orchestration/story (see scripts module)

Data Models / Schemas
- core/domain defines primary models; other modules consume them.

Notable Files
- {met['largest']}

Side Effects
- Network: aiohttp/requests in orchestration/story modules
- Storage: database adapters (psycopg2/oracledb), cache writes
- Filesystem: scene_bank JSON, config.yaml, POML templates

Feature Flags / Env Vars
- {', '.join(sorted(m['env_vars']))}

Complexity Signals
- Files: {met['files']}; LOC: {met['loc']}; TODO/FIXME: {met['todos']}

Suggested Refactors
- Stabilize imports via packaging; avoid sys.path hacks
- Encapsulate LLM providers behind interfaces; add timeouts/retries
- Harden secrets via vault/CI masks; validate DB config on startup
""".strip()
    with open(d/'SUMMARY.md','w',encoding='utf-8') as f: f.write(content)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--suffix', default='', help='Output suffix (e.g., L4) to avoid overwriting existing docs')
  ap.add_argument('--max-bytes', type=int, default=MAX_BYTES_DEFAULT)
  ap.add_argument('--root', default='.')
  args = ap.parse_args()
  modules, edges, env_vars, metrics = analyze(args.root, args.max_bytes)
  # If no suffix and files exist, switch to '.L4' to avoid overwrite
  suffix = args.suffix
  if not suffix and (os.path.exists('DOCS/REPO_MAP.md') or os.path.exists('DOCS/MODULES.md')):
    suffix = 'L4'
  write_docs(suffix, modules, edges, env_vars, metrics)
  print(f"Wrote docs with suffix: {suffix or '(none)'}")

if __name__ == '__main__':
  main()
