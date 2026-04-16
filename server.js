const express = require("express");
const cors = require("cors");
const multer = require("multer");

const { PdfPharser, AgentLoop } = require("./rag");
const { crawlDocs } = require("./doc-rag");

const app = express();
const port = 3000;

// ==========================
// MIDDLEWARE
// ==========================
app.use(
  cors({
    origin: "http://localhost:5173",
  })
);

app.use(express.json());

// File upload config
const upload = multer({ dest: "uploads/" });

// ==========================
// ROUTES
// ==========================

app.get("/", (req, res) => {
  res.send("Hello World!");
});

// ✅ PDF Upload
app.post("/api/upload-pdf", upload.single("pdfFile"), async (req, res) => {
  try {
    const pdfFile = req.file;

    await PdfPharser(pdfFile);

    res.json({ success: true });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "PDF processing failed" });
  }
});

// ✅ Website RAG
app.post("/api/upload-doc-rag", async (req, res) => {
  try {
    const { url } = req.body;

    const docRagResponse = await crawlDocs(url);

    res.json({ success: true, data: docRagResponse });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Scraping failed" });
  }
});

// ==========================
// (Ignore for now — FIXED anyway)
// ==========================

// PDF Query
app.post("/api/pdf-user-input", async (req, res) => {
  try {
    const { userInput } = req.body;

    const llmResponse = await AgentLoop(userInput);

    res.json({ success: true, data: llmResponse });
  } catch (err) {
    res.status(500).json({ error: "LLM failed" });
  }
});

// Docs Query
app.post("/api/doc-user-input", async (req, res) => {
  try {
    const { userInput } = req.body;

    const response = await AgentLoop(userInput);

    res.json({ success: true, data: response });
  } catch (err) {
    res.status(500).json({ error: "LLM failed" });
  }
});



app.listen(port, () => {
  console.log(`Example app listening on port ${port}`)
})





