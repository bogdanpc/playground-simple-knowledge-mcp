# DFF Knowledge Server

A simple MCP (Model Context Protocol) server provides Knowledge RAG over a local document collection.
Built application using [JBang](https://www.jbang.dev/), [LangChain4j MCP](https://docs.langchain4j.dev/tutorials/mcp/),
SQLite with vector search, and a local ONNX embedding model — no API keys required.

## Prerequisites

- **Java 25+**
- **JBang** — install via `curl -Ls https://sh.jbang.dev | bash` or `brew install jbang`

SQLite vector extension is downloaded on first run.

## Quick Start

1. Place your documents (PDF, TXT, etc.) in a `docs/` directory next to the script.

2. Run directly from GitHub (no clone needed):

   ```bash
   jbang KnowledgeMCP@bogdanpc/playground-simple-knowledge-mcp
   ```

   Or clone the repo and run locally:

   ```bash
   jbang KnowledgeMCP.java
   ```

On first run the server will:

- Download the `sqlite-vec` native extension for your platform
- Load and split all documents from `docs/`
- Generate embeddings using the local AllMiniLmL6V2 model
- Store everything in `knowledge.db`

Subsequent runs reuse the existing database and start immediately.

## Configuration

| Environment Variable | Default  | Description                     |
|----------------------|----------|---------------------------------|
| `DFF_DOCS_DIR`       | `./docs` | Path to the documents directory |

Example:

```bash
DFF_DOCS_DIR=/path/to/my/documents jbang KnowledgeMCP@bogdanpc/playground-simple-knowledge-mcp
```

## MCP Tools

The server exposes two tools over the MCP protocol (STDIO transport):

| Tool            | Description                                                                                             |
|-----------------|---------------------------------------------------------------------------------------------------------|
| `search`        | Semantic search over the knowledge base. Returns the top 5 most relevant text chunks for a given query. |
| `listDocuments` | Lists all PDF documents available in the docs directory.                                                |

## Using with Claude Code

Add the server to your Claude Code MCP configuration (`~/.claude/settings.json` or project `.claude/settings.json`):

```json
{
  "mcpServers": {
    "dff-knowledge": {
      "command": "jbang",
      "args": [
        "run",
        "KnowledgeMCP@bogdanpc/playground-simple-knowledge-mcp"
      ],
      "env": {
        "DFF_DOCS_DIR": "/absolute/path/to/docs"
      }
    }
  }
}
```

Once configured, Claude Code can call `search` and `listDocuments` to answer questions grounded in your documents.

## Using with Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "dff-knowledge": {
      "command": "jbang",
      "args": [
        "run",
        "KnowledgeMCP@bogdanpc/playground-simple-knowledge-mcp"
      ],
      "env": {
        "DFF_DOCS_DIR": "/absolute/path/to/docs"
      }
    }
  }
}
```

## Re-indexing

To re-index after adding or changing documents, delete the database and restart:

```bash
rm knowledge.db
jbang KnowledgeMCP@bogdanpc/playground-simple-knowledge-mcp
```

## How It Works

1. **Document loading** — LangChain4j's `FileSystemDocumentLoader` reads all files from the docs directory.
2. **Chunking** — Documents are split into ~500 character chunks with 50 character overlap using a recursive splitter.
3. **Embedding** — Each chunk is embedded locally using the AllMiniLmL6V2 ONNX model (384-dimensional vectors).
4. **Storage** — Chunks and embeddings are stored in SQLite using the `sqlite-vec` extension for vector similarity
   search.
5. **Retrieval** — On a `search` call, the query is embedded and the 5 nearest chunks are returned via vector distance
   matching.
6. **MCP transport** — The server communicates over STDIO, making it compatible with any MCP client.

## Supported Platforms

| Platform | Architecture                    |
|----------|---------------------------------|
| macOS    | aarch64 (Apple Silicon), x86_64 |
| Linux    | aarch64, x86_64                 |
| Windows  | x86_64                          |
