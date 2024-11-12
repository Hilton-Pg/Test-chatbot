import express from "express";
import path from "node:path";
import * as fs from "node:fs";
import multer from "multer";
import cors from "cors";
import dotenv from "dotenv";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RunnablePassthrough, RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";

dotenv.config(); // Load environment variables

const app = express();
const PORT = process.env.PORT || 4500;
app.use(cors());
app.use(express.json());

// Set up multer for file uploads
const upload = multer({ dest: "uploads/" });

app.get("/api/test",(req,res)=>{
  res.send("working")
})

// Endpoint to upload and process PDFs
app.post("/api/upload", upload.array("files", 10), async (req, res) => {
  try {
    const files = req.files;
    let allDocuments = [];

    // Load and process each uploaded PDF
    for (const file of files) {
      const loader = new PDFLoader(file.path);
      const docs = await loader.load();
      allDocuments.push(...docs);
    }

    // Split documents into smaller chunks
    const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
    const splitDocs = await textSplitter.splitDocuments(allDocuments);

    // Store documents in memory for retrieval
    const vectorStore = await MemoryVectorStore.fromDocuments(
      splitDocs,
      new OpenAIEmbeddings({ apiKey: process.env.OPENAI_API_KEY })
    );

    // Store the vector store for later use (in-memory only for this example)
    app.set("vectorStore", vectorStore);

    res.status(200).json({ message: "Files uploaded and processed successfully" });
  } catch (error) {
    console.error("Error processing PDFs:", error);
    res.status(500).json({ error: "Error processing PDFs" });
  }
});

// Endpoint to query the processed data
app.post("/api/ask", async (req, res) => {
  const { question } = req.body;
  const vectorStore = app.get("vectorStore");

  if (!vectorStore) {
    return res.status(400).json({ error: "No documents uploaded yet" });
  }

  try {
    const vectorStoreRetriever = vectorStore.asRetriever();

    // Create the system prompt
    const SYSTEM_TEMPLATE = `Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know; don't try to make up an answer.
    ----------------
    {context}`;

    const prompt = ChatPromptTemplate.fromMessages([
      ["system", SYSTEM_TEMPLATE],
      ["human", "{question}"],
    ]);

    const model = new ChatOpenAI({
      model: "gpt-4", // Specify the model you want to use
      apiKey: process.env.OPENAI_API_KEY,
    });

    const chain = RunnableSequence.from([
      {
        context: vectorStoreRetriever.pipe((docs) =>
          docs.map((d) => d.pageContent).join("\n\n")
        ),
        question: new RunnablePassthrough(),
      },
      prompt,
      model,
      new StringOutputParser(),
    ]);

    const answer = await chain.invoke(question);

    res.status(200).json({ answer });
  } catch (error) {
    console.error("Error processing query:", error);
    res.status(500).json({ error: "Error processing query" });
  }
});

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
