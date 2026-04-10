# 🏭 AI Content Factory (2026 Edition)

## 📌 Purpose
The **Content Factory** is an enterprise-grade, automated multi-agent system designed to generate highly engaging, short-form video content (Shorts/Reels/TikToks). 

Unlike standard AI video generators that hallucinate or rely on generic stock footage, this factory is purpose-built for **high-stakes domains**—such as global politics, macro-economics, and historical analysis. It treats **Truth and Guardrails as first-class citizens**, employing a rigorous "Red Team" agentic loop to verify claims against a vector database before any video or audio rendering occurs.

## 🚀 Core Differentiators
*   **Agentic Over Atomic:** Uses Plan-and-Execute workflows and Reflection Loops instead of linear pipelines. Agents debate and correct each other.
*   **Zero-Hallucination Guardrails:** Strict confidence constraints and multi-agent fact-checking ensure no false claims make it to the final render.
*   **Multi-Modal Native:** Bypasses generic stock footage by utilizing next-gen visual (Veo) and audio (Lyria) models, alongside precise Python-generated data visualization charts.
*   **Governance-as-Code:** Enforces SynthID watermarking compliance and platform safety guidelines natively within the PostgreSQL database structure.

---

## 🔄 The 8-Step Agentic Flow

The system operates asynchronously, moving a `RenderJob` through strict state transitions:

### 1. Ingestion (`PENDING`)
The user submits a high-level topic (e.g., *"BRICS De-dollarization 2025"*) along with "pre-context" (source URLs, PDF text, audience target) via the FastAPI endpoint.

### 2. Extraction & Structuring
A lightweight agent parses the pre-context, extracting key entities, timelines, and primary claims to guide the research phase.

### 3. Deep Research (`RESEARCHING`)
The **Research Agent** (powered by high-context models) queries the internal `pgvector` database for historical precedents and cross-references the user's pre-context. It generates highly structured `ResearchChunks`.

### 4. Source Fact-Check (`FACT_CHECKING_RESEARCH`)
An automated baseline check evaluates the `ResearchChunks` for credibility, dropping any unverified or low-confidence data points before they reach the writers' room.

### 5. Script & Storyboard Drafting (`SCRIPTING`)
The **Copywriter Agent** takes the verified research and drafts an engaging script optimized for short-form retention (AEO/SEO optimized). It also drafts a visual storyboard (prompts for the generation phase).

### 6. The Red Team Evaluator (`FACT_CHECKING_SCRIPT`)
*The most critical step.* The **Red Team Agent** breaks the script into atomic claims and verifies them against the original `ResearchChunks`. 
*   **If `SUPPORTED`:** The script passes.
*   **If `UNSUPPORTED`:** The script is rejected and sent back to Step 5 with feedback (*"Tone down Claim 3, it exaggerates the inflation rate"*). If it fails 3 times, it triggers `HUMAN_REVIEW_NEEDED`.

### 7. Multi-Modal Studio (`ASSET_GENERATION`)
Once the script is locked and verified, parallel agents generate the assets:
*   **Visuals:** Cinematic B-roll via video generation models (Veo) and data-accurate animated charts via Python.
*   **Audio:** High-fidelity Voiceover generation and dynamic background scoring (Lyria).
*   **Subtitles:** Word-level JSON timestamp generation.

### 8. Final Render & Delivery (`COMPLETED`)
Assets are stitched together, watermarked with SynthID for compliance, and uploaded to cloud storage. The FastAPI application returns the `final_video_url` and complete metadata payload.

---

## 🛠️ Tech Stack Overview
*   **API Framework:** FastAPI (Async, Pydantic V2 validation)
*   **Database:** PostgreSQL with `pgvector` (Strict typing, JSONB, GIN indexing)
*   **Migrations:** Alembic
*   **AI Orchestration:** Multi-Agent Python Workers (Celery/RQ) 
*   **Models:** Gemini 3.1 (Pro for Strategy/Copywriting, Flash for High-Volume RAG/Fact-checking)