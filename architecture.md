# ToolForge — Architecture

## Vue d'ensemble

ToolForge est un workspace Python (`uv`) composé de 8 packages interdépendants. L'idée centrale : un agent **Creator** reçoit une description en langage naturel d'un cas d'usage, conçoit des outils Python, les valide dans un sandbox, puis les expose via MCP. Un agent **Consumer** exécute ensuite des tâches en appelant ces outils.

```
┌──────────────────────────────────────────────────────────────────┐
│  toolforge CLI  (toolforge-cli)                                  │
│                                                                  │
│  toolforge tui ─────────────► TUI Selector                      │
│                                ├─ Creator screen                 │
│                                └─ Consumer screen                │
│                                                                  │
│  toolforge creator run ──► toolforge-mcp-creator (stdio)        │
│  │                          ├─ propose / validate / promote      │
│  │                          ├─ toolforge-registry (SQLite)       │
│  │                          └─ toolforge-sandbox  (uv | Docker)  │
│  │                                                               │
│  │   LLM ◄─── toolforge-core (Anthropic / OpenAI-compat)        │
│  └─ toolforge-tui (Textual)                                      │
│                                                                  │
│  toolforge consumer run ─► toolforge-mcp-usecase (stdio)        │
│                             ├─ call active tools                 │
│                             ├─ toolforge-registry                │
│                             └─ toolforge-sandbox                 │
│                                                                  │
│  Les deux serveurs écrivent dans toolforge-telemetry (JSONL)     │
└──────────────────────────────────────────────────────────────────┘
```

---

## Packages

### `toolforge-core`

Couche d'abstraction LLM partagée par tous les agents.

- **`LLMClient`** — interface unifiée pour Anthropic et les providers OpenAI-compatibles (Z.ai, OpenRouter, vLLM). La sélection du provider se fait via `toolforge.toml`.
- **`LLMAgent`** — boucle d'agent : envoie les messages, reçoit les events streamés (`TextDelta`, `ToolCallStart`, `ToolCallComplete`, `ToolResultEvent`, `MessageComplete`).
- **`CreatorAgent` / `ConsumerAgent`** — spécialisations de `LLMAgent` avec les context managers `creator_agent_stdio` / `consumer_agent_stdio` qui démarrent le serveur MCP sous-jacent en subprocess et connectent l'agent à ses tools.
- **`load_config`** — parse `toolforge.toml` (TOML natif Python 3.11+).

### `toolforge-registry`

Stockage des cas d'usage, runs et versions d'outils.

Structure sur disque :
```
data/
└── usecases/
    └── <usecase_id>/
        ├── usecase.json
        ├── prompt.md
        ├── inputs/          # fichiers d'entrée pour le Consumer
        ├── outputs/         # fichiers produits par les outils
        └── runs/
            └── <run_id>/
                ├── run.json
                ├── registry.db     # SQLite WAL
                ├── telemetry.jsonl
                └── tools/
                    └── <tool_name>/
                        ├── v1.py   # handler source
                        └── v1.json # JSON Schema des arguments
```

Un run passe par deux états : `draft` → `validated`. Un run validé est immutable ; pour itérer, on le **fork** (nouveau run draft qui hérite des outils).

### `toolforge-sandbox`

Exécution isolée du code handler. Deux modes :

| Mode | Isolation | Prérequis | Usage |
|---|---|---|---|
| `uv` (défaut) | Subprocess local | aucun | Développement |
| `docker` | Conteneur Docker | Docker daemon | Production / sécurité |

**Mode `uv`** : le runner est écrit une fois dans un fichier temporaire et réutilisé. Les dépendances sont installées via `uv run --with <req>` dans un cache éphémère. Aucun isolement réseau ou filesystem.

**Mode `docker`** : le runner est bake dans l'image (`/app/runner.py`). Seuls `/inputs` (ro) et `/outputs` (rw) sont montés. Le réseau bridge est autorisé (pour les appels LLM depuis les handlers). L'utilisateur dans le conteneur est non-root (`sandbox`, uid 1000).

Le `Dockerfile.sandbox` :
- Base : `python:3.12-slim`
- `uv` pré-installé (gestion de dépendances runtime)
- `runner.py` bake dans `/app/`
- Volumes déclarés : `/inputs`, `/outputs`

Configuration dans `toolforge.toml` :
```toml
[sandbox]
timeout_seconds = 30
mode            = "uv"                        # "uv" | "docker"
image           = "toolforge-sandbox:latest"  # utilisé seulement en mode docker
```

Construire l'image Docker :
```bash
uv run toolforge sandbox build
```

### `toolforge-mcp-creator`

Serveur MCP lancé en subprocess (stdio) par le CLI ou la TUI lors d'une session Creator. Expose les meta-outils à l'agent LLM :

| Outil MCP | Rôle |
|---|---|
| `propose_tool` | Déclare un nouvel outil (nom, description, JSON Schema, code handler) |
| `validate_in_sandbox` | Exécute le handler dans le sandbox avec des args de test |
| `promote_tool` | Promeut une version validée en version active |
| `deprecate_tool` | Désactive un outil |
| `request_human_validation` | Signale que le Creator a terminé et demande la relecture humaine |
| `read_telemetry` | Consulte les événements JSONL du run courant |

### `toolforge-mcp-usecase`

Serveur MCP lancé en subprocess (stdio) lors d'une session Consumer. Expose dynamiquement les outils actifs du run validé. Chaque outil correspond à un handler stocké dans le registry ; son exécution passe par le sandbox.

### `toolforge-tui`

Interface Textual (terminal UI). Trois écrans :

**`SelectorScreen`** — écran d'accueil, lancé par `toolforge tui`.
- Affiche l'arbre des cas d'usage et de leurs runs avec statut (`draft` / `validated`).
- Raccourcis clavier : `n` (nouveau cas d'usage), `v` (valider), `f` (fork), `e` (déverrouiller + éditer), `r` (rafraîchir).
- Palette de commandes (`Ctrl+P`) : Open Creator, Open Consumer, Validate, Fork, Edit run.

**`CreatorScreen`** — session interactive avec le Creator agent.
- Champ de saisie libre → l'agent répond en streaming.
- Les appels d'outils MCP apparaissent dans le log (`⚙ propose_tool`, `⚙ validate_in_sandbox`, …).
- `Escape` pour retourner au Selector.

**`ConsumerScreen`** — test d'un run validé.
- Panel principal : log de la conversation agent.
- Panel latéral tabulé : **Inputs** (fichiers disponibles, cliquer pour les injecter dans la tâche), **Calls** (historique des appels d'outils), **Outputs** (fichiers produits).
- `Escape` pour retourner au Selector.

### `toolforge-telemetry`

Log append-only en JSONL. Deux types d'événements :

```jsonl
{"kind":"creation","event":"propose_tool","tool":"<name>","version":1,"ts":"..."}
{"kind":"execution","event":"call_tool","tool":"<name>","duration_ms":42.1,"ts":"..."}
```

Option OpenTelemetry : activer `otel_enabled = true` dans `[telemetry]` et pointer `otel_endpoint` vers un collecteur (ex. `http://localhost:4318`).

### `toolforge-cli`

Point d'entrée `toolforge` (Click). Sous-commandes principales :

```
toolforge tui                          # TUI complète
toolforge usecase create / list
toolforge run create / list / validate / fork / tools
toolforge sandbox build
toolforge creator run
toolforge consumer run
```

---

## Graphe de dépendances inter-packages

```
toolforge-cli
├── toolforge-core          ← LLM, agents, config
│   └── mcp
├── toolforge-registry      ← filesystem + SQLite
├── toolforge-sandbox       ← exécution isolée
├── toolforge-mcp-creator
│   ├── toolforge-registry
│   ├── toolforge-sandbox
│   └── toolforge-telemetry
├── toolforge-mcp-usecase
│   ├── toolforge-registry
│   ├── toolforge-sandbox
│   └── toolforge-telemetry
├── toolforge-tui
│   └── toolforge-core
└── toolforge-telemetry     ← JSONL + OTel
```

---

## Flux de données : session Creator

```
User (TUI / CLI)
    │ message texte
    ▼
CreatorAgent (toolforge-core)
    │ messages + tool_use
    ▼
LLM (Anthropic / OpenAI-compat)
    │ tool_calls
    ▼
toolforge-mcp-creator (subprocess stdio)
    ├─ propose_tool    → toolforge-registry  (écrit v*.py + v*.json)
    ├─ validate        → toolforge-sandbox   (subprocess / docker run)
    │                        └─ runner.py ← handler code via stdin
    ├─ promote         → toolforge-registry  (active_version = N)
    └─ tous les events → toolforge-telemetry (append JSONL)
```

## Flux de données : session Consumer

```
User (TUI / CLI)
    │ tâche texte
    ▼
ConsumerAgent (toolforge-core)
    │ messages + tool_use
    ▼
LLM (Anthropic / OpenAI-compat)
    │ tool_calls
    ▼
toolforge-mcp-usecase (subprocess stdio)
    ├─ lit handler depuis toolforge-registry
    ├─ exécute via toolforge-sandbox
    │       └─ /inputs (ro) et /outputs (rw) montés en docker mode
    └─ écrit événements dans toolforge-telemetry
```

---

## Lancer le client TUI

```bash
# Installation
git clone <repo>
cd toolforge
uv sync --all-extras

# Configuration
cp toolforge.example.toml toolforge.toml
# Éditer toolforge.toml : provider LLM, mode sandbox, etc.

# (Optionnel) Construire l'image sandbox Docker
uv run toolforge sandbox build   # seulement si mode = "docker"

# Lancer la TUI complète
uv run toolforge tui

# Ou directement dans le Creator (deep-link)
run_id=$(uv run toolforge run create --usecase mon_cas)
uv run toolforge creator run --usecase mon_cas --run "$run_id"
```

Raccourcis TUI :
| Touche | Action |
|---|---|
| `Ctrl+P` | Ouvrir la palette de commandes |
| `n` | Nouveau cas d'usage |
| `v` | Valider le run sélectionné |
| `f` | Forker le run sélectionné |
| `e` | Déverrouiller et éditer le run |
| `r` | Rafraîchir la liste |
| `Escape` | Retour à l'écran précédent |
| `Ctrl+C` | Quitter |
