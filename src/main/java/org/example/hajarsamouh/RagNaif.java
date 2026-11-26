package org.example.hajarsamouh;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15q.BgeSmallEnV15QuantizedEmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.time.Duration;
import java.util.List;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Test 1 : RAG "naïf" avec les différentes étapes explicites.
 */
public class RagNaif {

    public static void main(String[] args) {
        configureLogger();

        // Le modèle de langage (Gemini), avec un délai plus long et quelques nouvelles tentatives
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(System.getenv("GEMINI_KEY"))
                .modelName("gemini-3.5-flash")
                .timeout(Duration.ofSeconds(120))
                .maxRetries(3)
                .logRequestsAndResponses(true)
                .build();

        // ===== Phase 1 : enregistrement des embeddings =====
        // 1. Document créé à partir du PDF (parseur Apache Tika)
        Document document = ClassPathDocumentLoader.loadDocument("rag.pdf", new ApacheTikaDocumentParser());

        // 2. Découpage en segments : 300 tokens maximum, 30 de chevauchement
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);

        // (Optionnel) pour voir les segments enregistrés, enlever les // de la ligne suivante :
        // segments.forEach(s -> System.out.println("--- SEGMENT ---\n" + s.text()));

        // 3. Modèle d'embedding
        EmbeddingModel embeddingModel = new BgeSmallEnV15QuantizedEmbeddingModel();

        // 4. Embeddings des segments (Response<List<Embedding>>, on prend le content())
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        // 5. Magasin d'embeddings en mémoire
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        // ===== Phase 2 : utilisation des embeddings pour répondre aux questions =====
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .contentRetriever(contentRetriever)
                .build();

        // Plusieurs questions sans recompiler ("fin" pour arrêter)
        conversationAvec(assistant);
    }

    /**
     * Permet de poser plusieurs questions à l'assistant sans recompiler.
     * Taper "fin" pour arrêter la conversation.
     *
     * @param assistant l'assistant avec lequel on dialogue.
     */
    private static void conversationAvec(Assistant assistant) {
        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.println("==================================================");
                System.out.println("Posez votre question : ");
                String question = scanner.nextLine();
                if (question.isBlank()) {
                    continue;
                }
                System.out.println("==================================================");
                if ("fin".equalsIgnoreCase(question)) {
                    break;
                }
                try {
                    String reponse = assistant.chat(question);
                    System.out.println("Assistant : " + reponse);
                } catch (Exception e) {
                    // Une erreur (timeout, 503...) n'arrête pas la conversation
                    System.out.println("Erreur : " + e.getMessage());
                }
                System.out.println("==================================================");
            }
        }
    }

    /**
     * Configure le logging de LangChain4j pour voir les requêtes et les réponses.
     */
    private static void configureLogger() {
        // Configure le logger sous-jacent (java.util.logging)
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE); // Ajuster niveau
        // Ajouter un handler pour la console pour faire afficher les logs
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }
}