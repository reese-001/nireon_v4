# NIREON Config Editor - Quick Start Guide

## Updated Implementation Summary

Based on the accepted improvements, here's the updated implementation plan:

### 🎯 Key Architectural Decisions

1. **Schema Single Source of Truth**: Schemas served from backend API
2. **Validation Reuse**: Using existing `NireonValidator` class
3. **Schema-First Editing**: Form is authoritative, YAML editing is secondary
4. **Git Backend**: All configs stored in Git for version control
5. **Performance Optimization**: Simple view filters, defer complex optimizations

## Quick Start (Development)

### 1. Backend Setup

```bash
# Clone NIREON repo (if not already available)
git clone <nireon-repo-url> nireon_v4

# Create config editor backend
mkdir nireon-config-editor/backend
cd nireon-config-editor/backend

# Create main.py (copy from artifact above)
# Create requirements.txt:
cat > requirements.txt << EOF
fastapi==0.104.1
uvicorn[standard]==0.24.0
pyyaml==6.0.1
jsonschema==4.20.0
python-multipart==0.0.6
pydantic==2.5.0
GitPython==3.1.40
EOF

# Install and run
pip install -r requirements.txt
PYTHONPATH=../nireon_v4 uvicorn main:app --reload
```

### 2. Frontend Setup

```bash
# Create React app with Vite
cd ..
npm create vite@latest frontend -- --template react-ts
cd frontend

# Install dependencies
npm install @mui/material @emotion/react @emotion/styled
npm install @rjsf/core @rjsf/mui @rjsf/validator-ajv8
npm install cytoscape @types/cytoscape
npm install js-yaml @types/js-yaml
npm install zustand react-query
npm install monaco-editor @monaco-editor/react

# Copy component files from artifacts above
# Start development server
npm run dev
```

### 3. Git Repository Setup

```bash
# Create a separate Git repo for configs
mkdir nireon_configs
cd nireon_configs
git init

# Copy existing configs
cp -r ../nireon_v4/configs .
git add .
git commit -m "Initial config import"
```

## Key Implementation Details

### Backend API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /api/schemas/{type}` | Fetch JSON schemas (single source of truth) |
| `POST /api/validate` | Validate using NIREON's validator |
| `POST /api/analyze-graph` | Generate graph data from rules |
| `POST /api/save-to-git` | Commit configs to Git |
| `GET /api/list-configs` | List available config files |
| `GET /api/load-config/{type}/{filename}` | Load specific config |

### Frontend Components

| Component | Purpose |
|-----------|---------|
| `RulesEditor` | Schema-driven form with autocomplete |
| `YamlPreview` | Monaco editor with schema-first mode |
| `GraphVisualizer` | Cytoscape with simplified view option |
| `GitControls` | Save to Git with branch/commit message |

### Performance Optimizations

1. **Graph Rendering**:
   - Show only enabled rules by default
   - Simplified view groups by namespace
   - Different layout algorithms based on size

2. **Form Performance**:
   - Component/signal autocomplete
   - Debounced validation
   - Lazy loading for large rule sets

### Security Considerations

For production deployment:

```python
# Update CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://nireon-config.yourcompany.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# Add authentication
from fastapi.security import HTTPBearer
security = HTTPBearer()

@app.post("/api/save-to-git")
async def save_to_git(
    config: ConfigData,
    commit_data: GitCommitData,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    # Verify JWT token
    verify_token(credentials.credentials)
    # ... rest of implementation
```

## CI/CD Integration

The GitHub Actions workflow validates configs on:
- Any PR that changes configs
- Schema changes (re-validates all)
- Validator script changes

## Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# With custom paths
NIREON_REPO_PATH=/path/to/nireon_v4 \
NIREON_CONFIG_REPO=/path/to/config/repo \
docker-compose up -d
```

## Testing the Implementation

1. **Unit Tests** (Frontend):
```typescript
// src/components/__tests__/RulesEditor.test.tsx
import { render, screen } from '@testing-library/react';
import { RulesEditor } from '../RulesEditor';

test('validates rules on change', async () => {
  render(<RulesEditor />);
  // ... test implementation
});
```

2. **Integration Tests** (Backend):
```python
# tests/test_api.py
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_schema_endpoint():
    response = client.get("/api/schemas/rules")
    assert response.status_code == 200
    assert "properties" in response.json()
```

## Monitoring & Maintenance

1. **Frontend Error Tracking**:
```typescript
// Add to App.tsx
import * as Sentry from "@sentry/react";
Sentry.init({ dsn: "YOUR_SENTRY_DSN" });
```

2. **Backend Metrics**:
```python
# Add to main.py
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)
```

## Next Steps

1. **Week 1**: Get basic form/graph working with real schemas
2. **Week 2**: Add Git integration and Monaco editor
3. **Week 3**: Performance optimizations and testing
4. **Week 4**: Security, monitoring, and production deployment

This implementation incorporates all accepted suggestions while keeping the MVP scope manageable.