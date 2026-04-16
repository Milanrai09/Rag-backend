import "dotenv/config"; // top of your entry file


// ================================================================
//  RAG Pipeline
//  Stack: OpenAI (embeddings + chat) · Pinecone (vector DB)
//         tiktoken (token-aware chunking)
//
//  Install dependencies:
//    npm install openai @pinecone-database/pinecone tiktoken uuid
//
//  Required env vars:
//    OPENAI_API_KEY
//    PINECONE_API_KEY
//    PINECONE_INDEX      — name of your Pinecone index
//    PINECONE_HOST       — full host URL from Pinecone console
//                         e.g. https://my-index-xxxx.svc.environment.pinecone.io
// ================================================================

import OpenAI from "openai";
import { Pinecone } from "@pinecone-database/pinecone";
import { encoding_for_model } from "tiktoken";
import { v4 as uuidv4 } from "uuid";
import fs from "fs";
import pdfParse from "pdf-parse";


// ─── Clients ────────────────────────────────────────────────
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const pineconeIndex = pinecone.index(
    process.env.PINECONE_INDEX,
    process.env.PINECONE_HOST
);

// ─── Config ──────────────────────────────────────────────────
const CONFIG = {
    embeddingModel:       "text-embedding-3-small",  // 1536-dim vectors
    chatModel:            "gpt-4o",
    chunkTokenSize:       256,    // max tokens per chunk
    chunkOverlapTokens:   32,     // overlap between adjacent chunks
    topK:                 5,      // chunks to retrieve per query
    maxAgentIterations:   5,
    pineconeNamespace:    "rag-default",
};


const PdfPharser = async (RAGdata) => {
    try {
        const dataBuffer = fs.readFileSync(RAGdata);
    
        const data = await pdfParse(dataBuffer);
    
        const text = data.text;
    
        return text; // ✅ IMPORTANT
      } catch (err) {
        console.error("PDF parsing failed:", err);
        throw err;
      }
}

// ─── 1. Data Chunking  (tiktoken-aware) ─────────────────────
/**
 * Splits raw text into token-bounded overlapping chunks.
 * Uses the same tokenizer as the embedding model so chunk
 * sizes are exact, not a character approximation.
 */
const DataChunking = async (RAGdata) => {
    const text = typeof RAGdata === "string" ? RAGdata : JSON.stringify(RAGdata);

    // cl100k_base is the tokenizer shared by text-embedding-3-* and gpt-4o
    const enc = encoding_for_model("text-embedding-3-small");
    const allTokens = enc.encode(text);

    const { chunkTokenSize, chunkOverlapTokens } = CONFIG;
    const chunks = [];
    let start = 0;

    while (start < allTokens.length) {
        const end = Math.min(start + chunkTokenSize, allTokens.length);
        // Decode this slice of token ids back to a UTF-8 string
        const sliceUint8 = enc.decode(allTokens.slice(start, end));
        chunks.push(new TextDecoder().decode(sliceUint8));
        start += chunkTokenSize - chunkOverlapTokens;
    }

    enc.free(); // release WASM memory

    console.log(`[DataChunking] ${chunks.length} chunks  |  ${allTokens.length} total tokens`);
    await Embedding(chunks);
};

// ─── 2. Chunk Embedding  (OpenAI) ───────────────────────────
/**
 * Sends all chunks to OpenAI in a single batched request.
 * OpenAI accepts up to 2 048 inputs per call.
 */
const Embedding = async (chunks) => {
    const response = await openai.embeddings.create({
        model:            CONFIG.embeddingModel,
        input:            chunks,
        encoding_format:  "float",
    });

    const embeddedData = chunks.map((chunk, i) => ({
        chunk,
        embedding: response.data[i].embedding,  // float32[1536]
    }));

    console.log(`[Embedding] Embedded ${embeddedData.length} chunks`);
    await StoreEmbedData(embeddedData);
};

// ─── 3. Store Embedded Data  (Pinecone) ─────────────────────
/**
 * Upserts all vectors into Pinecone.
 * The original chunk text is stored as metadata so retrieval
 * doesn't need a separate document store.
 */
const StoreEmbedData = async (embeddedData) => {
    const vectors = embeddedData.map(({ chunk, embedding }) => ({
        id:       uuidv4(),
        values:   embedding,
        metadata: { text: chunk },
    }));

    // Pinecone recommends batches of ≤ 100
    const BATCH_SIZE = 100;
    for (let i = 0; i < vectors.length; i += BATCH_SIZE) {
        await pineconeIndex
            .namespace(CONFIG.pineconeNamespace)
            .upsert(vectors.slice(i, i + BATCH_SIZE));
    }

    console.log(`[StoreEmbedData] Upserted ${vectors.length} vectors → Pinecone`);
};

// ─── 4. User Input Embedding ────────────────────────────────
/**
 * Embeds the user's question with the same model used for chunks
 * so cosine similarity scores are meaningful.
 */
const userInputEmbedding = async (userInput) => {
    const response = await openai.embeddings.create({
        model:            CONFIG.embeddingModel,
        input:            userInput,
        encoding_format:  "float",
    });
    return response.data[0].embedding;
};

// ─── 5. Input Query  (Pinecone ANN search) ──────────────────
/**
 * Runs an approximate-nearest-neighbour search in Pinecone and
 * returns the top-K most relevant chunk strings.
 */
const InputQuery = async (embedUserInput) => {
    const result = await pineconeIndex
        .namespace(CONFIG.pineconeNamespace)
        .query({
            vector:          embedUserInput,
            topK:            CONFIG.topK,
            includeMetadata: true,
        });

    const chunks = result.matches.map((m) => m.metadata.text);
    const topScores = result.matches
        .slice(0, 3)
        .map((m) => m.score.toFixed(3))
        .join(", ");

    console.log(`[InputQuery] ${chunks.length} chunks retrieved  |  top scores: ${topScores}`);
    return chunks;
};

// ─── 6. System Prompt ───────────────────────────────────────
const SystemPrompt = `
You are a RAG (Retrieval-Augmented Generation) assistant.
You will receive retrieved context chunks followed by the user's question.

Rules:
- Answer ONLY using the provided context chunks.
- If the context is insufficient, say so clearly instead of guessing.
- Be concise and accurate.
- When referencing information, cite the chunk, e.g. [Chunk 2].
`.trim();

// ─── 7. Agent Loop ──────────────────────────────────────────
/**
 * Full RAG query pipeline with a self-refinement loop.
 *
 * Flow per iteration:
 *   1st call  – answer from retrieved context
 *   2nd+ call – model sees its own previous answer and is asked
 *               to confirm or improve it (stops early if confident)
 */
async function AgentLoop(userInput) {
    // Retrieve relevant context
    const embedUserInput  = await userInputEmbedding(userInput);
    const contextChunks   = await InputQuery(embedUserInput);

    const contextBlock = contextChunks.length
        ? contextChunks.map((c, i) => `[Chunk ${i + 1}]\n${c}`).join("\n\n")
        : "(No relevant context found)";

    // Seed the conversation
    const messages = [
        { role: "system", content: SystemPrompt },
        {
            role:    "user",
            content: `CONTEXT:\n${contextBlock}\n\nQUESTION: ${userInput}`,
        },
    ];

    let finalAnswer = null;

    for (let i = 0; i < CONFIG.maxAgentIterations; i++) {
        console.log(`[AgentLoop] Iteration ${i + 1}`);

        const response = await openai.chat.completions.create({
            model:       CONFIG.chatModel,
            messages,
            temperature: 0.2,
        });

        const assistantMsg = response.choices[0].message;      // { role, content }
        const stopReason   = response.choices[0].finish_reason;
        const text         = assistantMsg.content ?? "";

        messages.push(assistantMsg);

        // Confidence check — exit if the model sounds certain
        const uncertain =
            text.toLowerCase().includes("i'm not sure") ||
            text.toLowerCase().includes("need more information") ||
            text.toLowerCase().includes("insufficient context");

        if (stopReason === "stop" && !uncertain) {
            finalAnswer = text;
            break;
        }

        // Ask the model to self-refine
        messages.push({
            role:    "user",
            content:
                "Review your previous answer against the context chunks. " +
                "If you can make it more accurate or complete, do so. " +
                "Otherwise confirm it as your final answer.",
        });
    }

    // Fallback: use the last assistant turn
    if (!finalAnswer) {
        const lastAssistant = [...messages]
            .reverse()
            .find((m) => m.role === "assistant");
        finalAnswer = lastAssistant?.content ?? "No answer generated.";
    }

    return finalAnswer;
}

// ─── Example usage ───────────────────────────────────────────
// import fs from "fs/promises";
// const doc = await fs.readFile("./knowledge-base.txt", "utf8");
// await Main({ RAGdata: doc, userInput: "What does the doc say about X?" });

export {
    Main,
    DataChunking,
    Embedding,
    StoreEmbedData,
    userInputEmbedding,
    InputQuery,
    AgentLoop,
    PdfPharser,
};