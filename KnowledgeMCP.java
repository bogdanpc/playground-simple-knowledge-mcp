/// usr/bin/env jbang "$0" "$@" ; exit $?
//DEPS dev.langchain4j:langchain4j:1.11.0
//DEPS dev.langchain4j:langchain4j-community-mcp-server:1.11.0-beta19
//DEPS dev.langchain4j:langchain4j-easy-rag:1.11.0-beta19
//DEPS dev.langchain4j:langchain4j-embeddings-all-minilm-l6-v2:1.11.0-beta19
//DEPS org.xerial:sqlite-jdbc:3.49.1.0
//DEPS org.slf4j:slf4j-simple:2.0.17
//JAVA_OPTIONS --enable-native-access=ALL-UNNAMED

//JAVA 25+

import dev.langchain4j.agent.tool.P;
import dev.langchain4j.agent.tool.Tool;
import dev.langchain4j.community.mcp.server.McpServer;
import dev.langchain4j.community.mcp.server.transport.StdioMcpServerTransport;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.mcp.protocol.McpImplementation;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.query.Query;
import org.sqlite.SQLiteConfig;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;

interface Log {
    static void info(String message) {
        LOG.log(System.Logger.Level.INFO, message);
    }
}

record Configuration(Path docsDir, Path dbPath) {

    static Configuration fromEnv() {
        var base = Path.of("");
        return new Configuration(
                Path.of(System.getenv().getOrDefault("DOCS_PATH", base.resolve("docs").toString())),
                Path.of(System.getenv().getOrDefault("DB_PATH", base.resolve("knowledge.db").toString()))
        );
    }

    Configuration {
        if (!Files.isDirectory(docsDir)) throw new IllegalArgumentException("Path " + docsDir + " is not a directory");
    }
}

static final System.Logger LOG = System.getLogger("KnowledgeServer");
static final Configuration CONFIG = Configuration.fromEnv();
static EmbeddingModel EMBEDDING_MODEL;
static Connection DB;

void main() throws Exception {
    // Configure JUL to write to stderr with a concise format
    // This is crucial for MCP servers over STDIO to avoid polluting stdout
    System.setProperty("java.util.logging.SimpleFormatter.format", "[%1$tT] %4$s: %5$s%6$s%n");

    // Redirect SLF4J to stderr so we don't corrupt the STDIO MCP protocol stream
    System.setProperty("org.slf4j.simpleLogger.logFile", "System.err");

    EMBEDDING_MODEL = new AllMiniLmL6V2EmbeddingModel();
    DB = createConnection(CONFIG.dbPath(), ensureVecExtension());
    ingest();

    var retriever = new SqliteContentRetriever(EMBEDDING_MODEL, DB);
    var tools = new KnowledgeTools(retriever, CONFIG.docsDir());

    var serverInfo = new McpImplementation();
    serverInfo.setName("knowledge-mcp-server");
    serverInfo.setVersion("1.0.0");

    var server = new McpServer(List.of(tools), serverInfo);
    new StdioMcpServerTransport(System.in, System.out, server);

    Log.info("MCP server ready. Waiting for requests on stdin...");

    // Keep the process alive while stdio is open
    Thread.currentThread().join();
}

/**
 * Ingestion documents from docs folder. Ingest only new files.
 */
void ingest() throws Exception {
    initializeSchema();

    // List files in docs directory
    List<Path> allFiles;
    try (var stream = Files.list(CONFIG.docsDir())) {
        allFiles = stream.filter(Files::isRegularFile).toList();
    }

    // Query ingested_files to find already-ingested ones
    var ingested = new HashSet<String>();
    try (var stmt = DB.createStatement();
         var rs = stmt.executeQuery("SELECT file_name FROM ingested_files")) {
        while (rs.next()) {
            ingested.add(rs.getString("file_name"));
        }
    }

    var newFiles = allFiles.stream()
            .filter(p -> !ingested.contains(p.getFileName().toString()))
            .toList();

    if (newFiles.isEmpty()) {
        Log.info("All files already ingested. Nothing to do.");
        return;
    }

    Log.info("Found " + newFiles.size() + " new file(s) to ingest.");
    ingestDocuments(newFiles);
}

enum Platform {
    MACOS_AARCH64("macos-aarch64", "dylib"),
    MACOS_X86_64("macos-x86_64", "dylib"),
    LINUX_AARCH64("linux-aarch64", "so"),
    LINUX_X86_64("linux-x86_64", "so"),
    WINDOWS_X86_64("windows-x86_64", "dll");

    final String classifier;
    final String ext;

    Platform(String classifier, String ext) {
        this.classifier = classifier;
        this.ext = ext;
    }

    static Platform detect() {
        var os = System.getProperty("os.name", "").toLowerCase();
        var arch = System.getProperty("os.arch", "").toLowerCase();
        if (os.contains("mac")) {
            return arch.contains("aarch64") || arch.contains("arm64") ? MACOS_AARCH64 : MACOS_X86_64;
        } else if (os.contains("win")) {
            return WINDOWS_X86_64;
        } else {
            return arch.contains("aarch64") || arch.contains("arm64") ? LINUX_AARCH64 : LINUX_X86_64;
        }
    }
}

String ensureVecExtension() throws Exception {
    var VEC_VERSION = "v0.1.7-alpha.8";

    var platform = Platform.detect();
    var extFileName = "vec0." + platform.ext;
    var extPath = Path.of(extFileName);

    if (Files.exists(extPath)) {
        Log.info("sqlite-vec extension found: " + extPath);
    } else {
        var tarName = "sqlite-vec-" + VEC_VERSION.substring(1) + "-loadable-" + platform.classifier + ".tar.gz";
        var url = "https://github.com/asg017/sqlite-vec/releases/download/" + VEC_VERSION + "/" + tarName;
        LOG.log(System.Logger.Level.INFO, "Downloading sqlite-vec from: " + url);

        try (var in = URI.create(url).toURL().openStream()) {
            extractFromTarGz(in, extFileName, extPath);
            LOG.log(System.Logger.Level.INFO, "Extracted: " + extPath);
        }

        if (!Files.exists(extPath)) {
            throw new RuntimeException("Failed to extract " + extFileName);
        }
    }

    // Return the load path (strip extension, add ./ prefix for relative)
    var loadName = extFileName.substring(0, extFileName.lastIndexOf('.'));
    return "./" + loadName;
}

/**
 * AI Generated
 * Extract a single file from a .tar.gz stream using only JDK APIs.
 * Tar format: 512-byte headers, filename at offset 0 (100 bytes), size at offset 124 (12 bytes, octal).
 * See <a href="https://en.wikipedia.org/wiki/Tar_(computing)#File_format">...</a>
 */
void extractFromTarGz(InputStream in, String targetName, Path targetPath) throws Exception {
    try (var gzip = new java.util.zip.GZIPInputStream(in);
         var buf = new BufferedInputStream(gzip)) {

        var header = new byte[512];
        while (buf.read(header) == 512) {
            if (header[0] == 0) break;
            var name = new String(header, 0, 100, StandardCharsets.US_ASCII).trim().replace("\0", "");
            var sizeStr = new String(header, 124, 12, StandardCharsets.US_ASCII).trim().replace("\0", "");
            var size = sizeStr.isEmpty() ? 0L : Long.parseLong(sizeStr, 8);

            if (name.endsWith(targetName)) {
                Files.copy(buf, targetPath, StandardCopyOption.REPLACE_EXISTING);
                return;
            } else {
                buf.skipNBytes(size);
                var pad = (512 - (size % 512)) % 512;
                if (pad > 0) buf.skipNBytes(pad);
            }
        }
    }
    throw new RuntimeException("Entry " + targetName + " not found in tar archive");
}

/**
 * Create SQLite database connection with vector extension
 */
Connection createConnection(Path dbPath, String extLoadPath) throws Exception {
    var config = new SQLiteConfig();
    config.enableLoadExtension(true);
    var conn = DriverManager.getConnection("jdbc:sqlite:" + dbPath, config.toProperties());

    try (var stmt = conn.createStatement()) {
        stmt.execute("SELECT load_extension('" + extLoadPath + "')");
    }
    return conn;
}


void initializeSchema() throws Exception {
    try (var stmt = DB.createStatement()) {
        stmt.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        text TEXT NOT NULL,
                        file_name TEXT
                    )
                """);
        stmt.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS vec_documents USING vec0(
                        embedding float[384]
                    )
                """);
        stmt.execute("""
                    CREATE TABLE IF NOT EXISTS ingested_files (
                        file_name TEXT PRIMARY KEY,
                        ingested_at TEXT DEFAULT (datetime('now'))
                    )
                """);
    }
}

void ingestDocuments(List<Path> files) throws Exception {
    var documents = new ArrayList<Document>();
    for (var file : files) {
        documents.add(FileSystemDocumentLoader.loadDocument(file));
    }

    var splitter = DocumentSplitters.recursive(500, 50);
    var segments = new ArrayList<TextSegment>();
    for (var doc : documents) {
        segments.addAll(splitter.split(doc));
    }

    var embeddings = EMBEDDING_MODEL.embedAll(segments).content();
    Log.info("Generated " + embeddings.size() + " embeddings.");

    DB.setAutoCommit(false);
    try (var insertDoc = DB.prepareStatement(
                 "INSERT INTO documents (text, file_name) VALUES (?, ?)",
                 Statement.RETURN_GENERATED_KEYS);
         var insertVec = DB.prepareStatement(
                 "INSERT INTO vec_documents (rowid, embedding) VALUES (?, ?)");
         var insertIngested = DB.prepareStatement(
                 "INSERT INTO ingested_files (file_name) VALUES (?)")) {

        for (int i = 0; i < segments.size(); i++) {
            var segment = segments.get(i);
            var embedding = embeddings.get(i);

            var fileName = segment.metadata() != null ? segment.metadata().getString("file_name") : null;
            insertDoc.setString(1, segment.text());
            insertDoc.setString(2, fileName);
            insertDoc.executeUpdate();

            try (var keys = insertDoc.getGeneratedKeys()) {
                keys.next();
                var rowId = keys.getLong(1);
                insertVec.setLong(1, rowId);
                insertVec.setBytes(2, floatsToBytes(embedding.vector()));
                insertVec.executeUpdate();
            }
        }

        for (var file : files) {
            insertIngested.setString(1, file.getFileName().toString());
            insertIngested.addBatch();
        }
        insertIngested.executeBatch();

        DB.commit();
    } finally {
        DB.setAutoCommit(true);
    }
    Log.info("Ingestion complete. " + segments.size() + " chunks stored in " + CONFIG.dbPath());
}

/**
 * SQLite Content Retriever
 */
class SqliteContentRetriever implements ContentRetriever {

    private final EmbeddingModel embeddingModel;
    private final Connection conn;

    SqliteContentRetriever(EmbeddingModel embeddingModel, Connection conn) {
        this.embeddingModel = embeddingModel;
        this.conn = conn;
    }

    @Override
    public List<Content> retrieve(Query query) {
        try {
            var queryEmbedding = embeddingModel.embed(query.text()).content();

            var sql = """
                        SELECT d.text, d.file_name, v.distance
                        FROM vec_documents v
                        JOIN documents d ON d.id = v.rowid
                        WHERE v.embedding MATCH ? AND k = 5
                        ORDER BY v.distance
                    """;

            var results = new ArrayList<Content>();
            try (var ps = conn.prepareStatement(sql)) {
                ps.setBytes(1, floatsToBytes(queryEmbedding.vector()));
                var rs = ps.executeQuery();

                while (rs.next()) {
                    var text = rs.getString("text");
                    var fileName = rs.getString("file_name");
                    var metadata = new Metadata();
                    if (fileName != null) {
                        metadata.put("file_name", fileName);
                    }
                    results.add(Content.from(TextSegment.from(text, metadata)));
                }
            }

            return results;
        } catch (SQLException e) {
            System.err.println("SQLite search failed: " + e.getMessage());
            return List.of();
        }
    }
}

static byte[] floatsToBytes(float[] floats) {
    var buf = ByteBuffer.allocate(floats.length * 4).order(ByteOrder.LITTLE_ENDIAN);
    for (var f : floats) buf.putFloat(f);
    return buf.array();
}

class KnowledgeTools {
    private final ContentRetriever retriever;
    private final Path docsDirectory;

    KnowledgeTools(ContentRetriever retriever, Path docsDirectory) {
        this.retriever = retriever;
        this.docsDirectory = docsDirectory;
    }

    @Tool("Search knowledge base for relevant information. Returns the most relevant text chunks matching the query.")
    public String search(@P("The search query describing what information you need") String query) {
        var results = retriever.retrieve(Query.from(query));

        if (results.isEmpty()) {
            return "No relevant results found for: " + query;
        }

        var sb = new StringBuilder();
        sb.append("Found ").append(results.size()).append(" relevant chunks:\n\n");

        for (int i = 0; i < results.size(); i++) {
            var content = results.get(i);
            var segment = content.textSegment();
            sb.append("--- Result ").append(i + 1).append(" ---\n");

            if (segment.metadata() != null && segment.metadata().getString("file_name") != null) {
                sb.append("Source: ").append(segment.metadata().getString("file_name")).append("\n");
            }

            sb.append(segment.text()).append("\n\n");
        }

        return sb.toString();
    }

    @Tool("List all documents available in the knowledge base")
    public String listDocuments() {

        try (var walk = Files.walk(docsDirectory)) {
            return walk
                    .filter(Files::isRegularFile)
                    .map(Path::getFileName)
                    .map(Path::toString)
                    .sorted()
                    .collect(Collectors.joining("\n"));
        } catch (IOException e) {
            System.err.println("Failed to list documents in " + docsDirectory);
            return "Documents directory not found.";
        }
    }
}
