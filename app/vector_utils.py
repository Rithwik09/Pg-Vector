from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple
import nltk
import re
from collections import namedtuple

# Download required NLTK resources
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

# Define a document chunk structure
DocumentChunk = namedtuple('DocumentChunk', 
                          ['content', 'embedding', 'metadata'])

class DocumentProcessor:
    def __init__(self, model_name: str = "multi-qa-mpnet-base-dot-v1"):
        """Initialize with a sentence transformer model."""
        self.model = SentenceTransformer(model_name)
        
    def preprocess_text(self, text: str) -> str:
        """Clean extracted PDF text by removing artifacts and normalizing whitespace."""
        # Remove page numbers and headers/footers
        text = re.sub(r'\n\d+\s*\n', '\n', text)
        text = re.sub(r'\nO\s*HENRY\s*-\s*100\s*SELECTED\s*STORIES\s*\n', '\n', text)
        
        # Normalize whitespace - convert multiple spaces to single space
        text = re.sub(r'\s+', ' ', text)
        
        # Fix broken words at line breaks (words with a hyphen at line end)
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # Normalize paragraph breaks (double newlines)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip() 
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Generate normalized embedding for text."""
        if not text.strip():
            return np.zeros(self.model.get_sentence_embedding_dimension())
            
        # embedding = self.model.encode(text)
        embedding = self.model.encode(text, normalize_embeddings=False)
        norm = np.linalg.norm(embedding)
        
        if norm == 0 or not np.isfinite(norm):
            return np.zeros(len(embedding))
            
        return embedding / norm
    
    def split_text_into_chunks(self, 
                               text: str, 
                               source_name: str = "",
                               max_tokens: int = 300, 
                               overlap_tokens: int = 50) -> List[DocumentChunk]:
        """
        Split text into overlapping chunks with metadata and generate embeddings.
        
        Args:
            text: The document text to chunk
            source_name: Name of the source document
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Number of tokens to overlap between chunks
            
        Returns:
            List of DocumentChunk objects with content, embedding, and metadata
        """
        # Clean the text
        text = self.preprocess_text(text)
        
        # Split into sentences
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = sentence.split()
            token_count = len(sentence_tokens)
            
            # If adding this sentence exceeds max_tokens, finalize the chunk
            if current_length + token_count > max_tokens and current_chunk:
                # Create the chunk content
                chunk_text = " ".join(current_chunk)
                
                # Create metadata
                metadata = {
                    "source": source_name,
                    "chunk_index": chunk_index,
                    "sentence_range": f"{i-len(current_chunk)}-{i-1}",
                    "token_count": current_length
                }
                
                # Generate embedding
                embedding = self.get_embedding(chunk_text)
                
                # Create chunk object
                chunks.append(DocumentChunk(chunk_text, embedding, metadata))
                
                # For overlap, keep some sentences from the end
                overlap_sentences = []
                overlap_token_count = 0
                
                # Add sentences from the end until we reach overlap_tokens
                for sent in reversed(current_chunk):
                    sent_tokens = len(sent.split())
                    if overlap_token_count + sent_tokens <= overlap_tokens:
                        overlap_sentences.insert(0, sent)
                        overlap_token_count += sent_tokens
                    else:
                        break
                
                # Start new chunk with overlap sentences
                current_chunk = overlap_sentences
                current_length = overlap_token_count
                chunk_index += 1
            
            # Add the current sentence to the chunk
            current_chunk.append(sentence)
            current_length += token_count
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            metadata = {
                "source": source_name,
                "chunk_index": chunk_index,
                "sentence_range": f"{len(sentences)-len(current_chunk)}-{len(sentences)-1}",
                "token_count": current_length
            }
            embedding = self.get_embedding(chunk_text)
            chunks.append(DocumentChunk(chunk_text, embedding, metadata))
        
        print(f"Total Chunks: {len(chunks)}")
        return chunks
    
    def find_relevant_chunks(self, 
                             query: str, 
                             chunks: List[DocumentChunk], 
                             top_k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        """
        Find most relevant chunks for a query using semantic search.
        
        Args:
            query: The search query
            chunks: List of DocumentChunk objects
            top_k: Number of top results to return
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        query_embedding = self.get_embedding(query)
        
        # Calculate similarities
        similarities = []
        for chunk in chunks:
            if np.all(chunk.embedding == 0):
                print(f"Warning: Chunk {chunk.metadata['chunk_index']} has zero embedding.")
            # Calculate cosine similarity
            print("Query embedding:", query_embedding)
            print("Chunk embedding:", chunk.embedding)
            similarity = np.dot(query_embedding, chunk.embedding)
            print("Dot product similarity:", similarity)
            similarities.append((chunk, similarity))
        
        # Sort by similarity (descending)
        sorted_results = sorted(similarities, key=lambda x: x[1], reverse=True)
        
        # Return top k results
        return sorted_results[:top_k]

    def process_document(self, 
                        text: str, 
                        source_name: str,
                        max_tokens: int = 300, 
                        overlap_tokens: int = 50) -> List[DocumentChunk]:
        """Process a document from start to finish."""
        return self.split_text_into_chunks(
            text=text,
            source_name=source_name,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens
        )