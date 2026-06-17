-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents metadata table
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL,
    source_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create document chunks table for hybrid retrieval
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(768) NOT NULL, -- 768 dimensions for Nomic / Gemma Embeddings
    allowed_roles VARCHAR(50)[] NOT NULL,
    allowed_jurisdictions VARCHAR(50)[] NOT NULL,
    chunk_index INT NOT NULL,
    page_number INT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create B-Tree index on document_id for joins
CREATE INDEX IF NOT EXISTS chunk_doc_id_idx ON document_chunks(document_id);

-- Create HNSW index for cosine distance vector operations
-- Note: ef_construction=64 and m=16 are safe local defaults for fast build/query tradeoffs.
CREATE INDEX IF NOT EXISTS chunk_hnsw_idx ON document_chunks 
USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- Create GIN indexes for fast array matching on roles and jurisdictions
CREATE INDEX IF NOT EXISTS chunk_roles_idx ON document_chunks USING gin (allowed_roles);
CREATE INDEX IF NOT EXISTS chunk_jurisdictions_idx ON document_chunks USING gin (allowed_jurisdictions);
