# Model Update Workflow - Decentralized Architecture

## Overview

FinDash uses a decentralized model update system where models are uploaded to IPFS/cloud storage after training, and clients detect and download updates.

---

## Workflow

### 1. Model Training & Upload

```
Train Model → Test & Validate → Upload to IPFS/Cloud → Get CID/URL → Update Registry
```

**Process**:
1. Train model locally or on compute cluster
2. Validate model performance
3. Upload model files to IPFS or cloud storage
4. Get Content ID (CID) or download URL
5. Update model registry with new CID/URL

### 2. Model Registry Update

**Registry Location**: Could be:
- Smart contract on blockchain
- Centralized API endpoint
- IPFS-hosted JSON manifest
- P2P network announcement

**Registry Format**:
```json
{
  "models": {
    "embeddings": {
      "version": "1.0.2",
      "cid": "QmXxx...",
      "url": "https://storage.example.com/models/embeddings-v1.0.2.tar.gz",
      "hash": "sha256:abc123...",
      "size_mb": 90,
      "updated_at": "2025-12-13T10:00:00Z"
    },
    "sentiment": {
      "version": "2.1.0",
      "cid": "QmYyy...",
      "url": "https://storage.example.com/models/sentiment-v2.1.0.tar.gz",
      "hash": "sha256:def456...",
      "size_mb": 420,
      "updated_at": "2025-12-12T15:30:00Z"
    }
  }
}
```

### 3. Client Detection

**Client checks for updates**:
- On app startup
- Periodically (e.g., every 24 hours)
- Manual check via UI button

**Detection Process**:
```typescript
async checkForModelUpdates() {
  // Fetch latest registry
  const registry = await fetch('/api/models/registry').then(r => r.json());
  
  // Compare with local versions
  const localVersions = await getLocalModelVersions();
  
  // Find updates
  const updates = [];
  for (const [modelName, modelInfo] of Object.entries(registry.models)) {
    if (modelInfo.version > localVersions[modelName]) {
      updates.push({ modelName, ...modelInfo });
    }
  }
  
  return updates;
}
```

### 4. User Prompt

**When updates detected**:
```
┌─────────────────────────────────────────────┐
│  🔄 Model Updates Available                 │
├─────────────────────────────────────────────┤
│  • Embeddings v1.0.2 (90 MB)                │
│    Improved semantic search accuracy        │
│                                             │
│  • Sentiment v2.1.0 (420 MB)                │
│    Better financial sentiment detection     │
│                                             │
│  [Download Now]  [Remind Me Later]  [Skip] │
└─────────────────────────────────────────────┘
```

### 5. Model Download

**Download Process**:
```typescript
async downloadModel(modelInfo) {
  // Show progress
  const progress = new ProgressTracker();
  
  // Download from IPFS or cloud
  const source = modelInfo.cid 
    ? `ipfs://${modelInfo.cid}`
    : modelInfo.url;
  
  const modelData = await downloadWithProgress(source, progress);
  
  // Verify hash
  const hash = await calculateHash(modelData);
  if (hash !== modelInfo.hash) {
    throw new Error('Model hash mismatch');
  }
  
  // Extract and save to /models/llm/
  await extractModel(modelData, `/models/llm/${modelInfo.path}`);
  
  // Update local version registry
  await updateLocalVersion(modelInfo.name, modelInfo.version);
  
  // Reload services
  await reloadModelServices();
}
```

---

## Current Architecture

### Backend Models (ServerBackend)

**Location**: `/models/llm/`

**Usage**: Backend services load models from local directory
- Embedding service: `/models/llm/embeddings/all-MiniLM-L6-v2`
- Sentiment service: `/models/llm/sentiment/finbert`
- LLM service: `/models/llm/llm/phi-3-mini-4k-instruct`

**Loading**:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("/models/llm/embeddings/all-MiniLM-L6-v2")
```

### Frontend Models (DISABLED)

**Previous Approach**: Download models in browser using Transformers.js
**Issue**: Large downloads, HuggingFace dependency, browser memory limits

**New Approach**: Use backend API for LLM inference
```typescript
// Instead of loading models in browser
async function generateText(prompt: string) {
  const response = await fetch('/api/llm/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, max_tokens: 512 })
  });
  return response.json();
}
```

---

## Implementation Plan

### Phase 1: Fix Current Errors ✅

- [x] Disable browser-based model loading in `transformers-engine.ts`
- [x] Update embedding service to use local models
- [x] Document architecture

### Phase 2: Backend API for LLM

- [ ] Create `/api/llm/generate` endpoint
- [ ] Load Phi-3 model on backend
- [ ] Implement streaming responses
- [ ] Add rate limiting

### Phase 3: Model Registry

- [ ] Create model registry API
- [ ] Store model versions in database
- [ ] Implement version comparison

### Phase 4: Update Detection

- [ ] Frontend checks for updates on startup
- [ ] Show update notification UI
- [ ] Implement download with progress

### Phase 5: IPFS/Cloud Download

- [ ] Support IPFS CID downloads
- [ ] Support cloud storage URLs
- [ ] Verify model hashes
- [ ] Extract and install models

---

## API Endpoints (To Implement)

### Model Registry

```
GET /api/models/registry
Response: {
  models: {
    embeddings: { version, cid, url, hash, size_mb },
    sentiment: { version, cid, url, hash, size_mb },
    llm: { version, cid, url, hash, size_mb }
  }
}
```

### Model Download

```
POST /api/models/download
Body: { model_name: string, source: string }
Response: { task_id: string }

GET /api/models/download/{task_id}/status
Response: { progress: number, status: string }
```

### LLM Generation

```
POST /api/llm/generate
Body: { prompt: string, max_tokens: number, temperature: number }
Response: { text: string, tokens_used: number }

POST /api/llm/generate/stream (SSE)
Body: { prompt: string }
Response: Server-Sent Events stream
```

---

## Next Steps

1. ✅ Fix immediate Transformers.js error
2. ✅ Update embedding service to use local models
3. Create backend LLM API endpoint
4. Implement model registry
5. Build update detection UI
6. Add IPFS/cloud download support
