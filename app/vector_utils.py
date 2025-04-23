# from sentence_transformers import SentenceTransformer
# import numpy as np
# from typing import List, Tuple
# import nltk
# import re
# from collections import namedtuple

# nltk.download('punkt', quiet=True)
# from nltk.tokenize import sent_tokenize

# DocumentChunk = namedtuple('DocumentChunk', ['content', 'embedding', 'metadata'])

# class DocumentProcessor:
#     def __init__(self, model_name: str = "multi-qa-mpnet-base-dot-v1"):
#         """Initialize with a sentence transformer model (768-dimensional)."""
#         self.model = SentenceTransformer(model_name)
#         self.embedding_dim = self.model.get_sentence_embedding_dimension()
#         print(f"✅ Model loaded: {model_name}, Embedding Dimension: {self.embedding_dim}")

#     def preprocess_text(self, text: str) -> str:
#         """Clean extracted PDF text by removing artifacts and normalizing whitespace."""
#         text = re.sub(r'\n\d+\s*\n', '\n', text)
#         text = re.sub(r'\nO\s*HENRY\s*-\s*100\s*SELECTED\s*STORIES\s*\n', '\n', text)
#         text = re.sub(r'\s+', ' ', text)
#         text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
#         text = re.sub(r'\n\s*\n', '\n\n', text)
#         return text.strip()

#     def get_embedding(self, text: str) -> np.ndarray:
#         """Generate normalized 768-dimensional embedding for text."""
#         if not text.strip():
#             return np.zeros(self.embedding_dim)
        
#         embedding = self.model.encode(text, normalize_embeddings=False)
#         norm = np.linalg.norm(embedding)
#         if norm == 0 or not np.isfinite(norm):
#             return np.zeros(len(embedding))
#         return embedding / norm

#     def split_text_into_chunks(self, 
#                                text: str, 
#                                source_name: str = "",
#                                max_tokens: int = 300, 
#                                overlap_tokens: int = 50) -> List[DocumentChunk]:
#         """Split text into overlapping chunks with 768-dim embeddings."""
#         text = self.preprocess_text(text)
#         sentences = sent_tokenize(text)
        
#         chunks = []
#         current_chunk = []
#         current_length = 0
#         chunk_index = 0
        
#         for i, sentence in enumerate(sentences):
#             sentence_tokens = sentence.split()
#             token_count = len(sentence_tokens)
            
#             if current_length + token_count > max_tokens and current_chunk:
#                 chunk_text = " ".join(current_chunk)
#                 metadata = {
#                     "source": source_name,
#                     "chunk_index": chunk_index,
#                     "sentence_range": f"{i-len(current_chunk)}-{i-1}",
#                     "token_count": current_length
#                 }
#                 embedding = self.get_embedding(chunk_text)
#                 chunks.append(DocumentChunk(chunk_text, embedding, metadata))
                
#                 # Overlap logic
#                 overlap_sentences = []
#                 overlap_token_count = 0
#                 for sent in reversed(current_chunk):
#                     sent_tokens = len(sent.split())
#                     if overlap_token_count + sent_tokens <= overlap_tokens:
#                         overlap_sentences.insert(0, sent)
#                         overlap_token_count += sent_tokens
#                     else:
#                         break
                
#                 current_chunk = overlap_sentences
#                 current_length = overlap_token_count
#                 chunk_index += 1
            
#             current_chunk.append(sentence)
#             current_length += token_count
        
#         if current_chunk:
#             chunk_text = " ".join(current_chunk)
#             metadata = {
#                 "source": source_name,
#                 "chunk_index": chunk_index,
#                 "sentence_range": f"{len(sentences)-len(current_chunk)}-{len(sentences)-1}",
#                 "token_count": current_length
#             }
#             embedding = self.get_embedding(chunk_text)
#             chunks.append(DocumentChunk(chunk_text, embedding, metadata))
        
#         print(f"📄 Total Chunks: {len(chunks)}")
#         return chunks

#     def find_relevant_chunks(self, 
#                              query: str, 
#                              chunks: List[DocumentChunk], 
#                              top_k: int = 3) -> List[Tuple[DocumentChunk, float]]:
#         """Find most relevant chunks for a query using 768-dim cosine similarity."""
#         query_embedding = self.get_embedding(query)
        
#         similarities = []
#         for chunk in chunks:
#             if np.all(chunk.embedding == 0):
#                 print(f"⚠️ Warning: Chunk {chunk.metadata['chunk_index']} has zero embedding.")
#             similarity = np.dot(query_embedding, chunk.embedding)
#             similarities.append((chunk, similarity))
        
#         sorted_results = sorted(similarities, key=lambda x: x[1], reverse=True)
#         return sorted_results[:top_k]

#     def process_document(self, 
#                          text: str, 
#                          source_name: str,
#                          max_tokens: int = 300, 
#                          overlap_tokens: int = 50) -> List[DocumentChunk]:
#         """Process a document with 768-dim embeddings from start to finish."""
#         return self.split_text_into_chunks(
#             text=text,
#             source_name=source_name,
#             max_tokens=max_tokens,
#             overlap_tokens=overlap_tokens
#         )


# from sentence_transformers import SentenceTransformer
# import numpy as np
# from typing import List, Tuple
# import nltk
# import re
# from collections import namedtuple

# nltk.download('punkt', quiet=True)
# from nltk.tokenize import sent_tokenize

# DocumentChunk = namedtuple('DocumentChunk', ['content', 'embedding', 'metadata'])

# class DocumentProcessor:
#     def __init__(self, model_name: str = "multi-qa-mpnet-base-dot-v1"):
#         self.model = SentenceTransformer(model_name)
#         self.embedding_dim = self.model.get_sentence_embedding_dimension()
#         print(f"✅ Model loaded: {model_name}, Embedding Dimension: {self.embedding_dim}")

#     def preprocess_text(self, text: str) -> str:
#         """Clean text while preserving important phrases"""
#         text = re.sub(r'\n\d+\s*\n', '\n', text)  # Remove page numbers
#         text = re.sub(r'\s+', ' ', text)          # Normalize whitespace
#         text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)  # Fix hyphenated words
#         return text.strip()

#     def get_embedding(self, text: str) -> np.ndarray:
#         """Generate normalized embeddings using model's native normalization"""
#         if not text.strip():
#             return np.zeros(self.embedding_dim)
#         return self.model.encode(text, normalize_embeddings=True)

#     def split_text_into_chunks(
#         self,
#         text: str,
#         source_name: str = "",
#         max_tokens: int = 100,
#         overlap_sentences: int = 1
#     ) -> List[DocumentChunk]:
#         """Sentence-aware chunking with context overlap"""
#         text = self.preprocess_text(text)
#         sentences = sent_tokenize(text)
        
#         chunks = []
#         current_chunk = []
#         current_token_count = 0
        
#         for i, sentence in enumerate(sentences):
#             sentence_tokens = sentence.split()
#             sentence_length = len(sentence_tokens)
            
#             if current_token_count + sentence_length > max_tokens:
#                 if current_chunk:  # Finalize current chunk
#                     chunks.append(" ".join(current_chunk))
#                     # Carry over overlapping sentences
#                     current_chunk = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
#                     current_token_count = sum(len(s.split()) for s in current_chunk)
                
#             current_chunk.append(sentence)
#             current_token_count += sentence_length
        
#         # Add remaining content
#         if current_chunk:
#             chunks.append(" ".join(current_chunk))
        
#         # Create DocumentChunk objects
#         document_chunks = []
#         for chunk_idx, chunk_text in enumerate(chunks):
#             metadata = {
#                 "source": source_name,
#                 "chunk_index": chunk_idx,
#                 "sentences": len(sent_tokenize(chunk_text)),
#                 "token_count": len(chunk_text.split())
#             }
#             embedding = self.get_embedding(chunk_text)
#             document_chunks.append(DocumentChunk(chunk_text, embedding, metadata))
        
#         print(f"📄 Generated {len(document_chunks)} chunks with sentence-aware splitting")
#         return document_chunks

#     # def find_relevant_chunks(
#     #     self,
#     #     query: str,
#     #     chunks: List[DocumentChunk],
#     #     top_k: int = 3
#     # ) -> List[Tuple[DocumentChunk, float]]:
#     #     """Semantic search with enhanced scoring"""
#     #     query_embedding = self.get_embedding(query)
        
#     #     # Calculate cosine similarities
#     #     similarities = []
#     #     for chunk in chunks:
#     #         if chunk.embedding.shape[0] != self.embedding_dim:
#     #             print(f"⚠️ Dimension mismatch in chunk {chunk.metadata['chunk_index']}")
#     #             continue
                
#     #         similarity = np.dot(query_embedding, chunk.embedding)
#     #         similarities.append((chunk, similarity))
        
#     #     # Sort and filter top results
#     #     sorted_results = sorted(similarities, key=lambda x: x[1], reverse=True)
#     #     return sorted_results[:top_k]

# def find_relevant_chunks(
#     self,
#     query: str,
#     chunks: List[DocumentChunk],
#     top_k: int = 3
# ):
#     query_embedding = self.get_embedding(query)
    
#     similarities = []
#     for chunk in chunks:
#         similarity = np.dot(query_embedding, chunk.embedding)
        
#         # Boost score for exact matches (optional)
#         if query.lower() in chunk.content.lower():
#             similarity += 0.3  # Boost factor
            
#         similarities.append((chunk, similarity))
    
#     sorted_results = sorted(similarities, key=lambda x: x[1], reverse=True)
#     return sorted_results[:top_k]

# def process_document(
#         self,
#         text: str,
#         source_name: str,
#         max_tokens: int = 100,
#         overlap_sentences: int = 1
#     ) -> List[DocumentChunk]:
#         """End-to-end document processing pipeline"""
#         return self.split_text_into_chunks(
#             text=text,
#             source_name=source_name,
#             max_tokens=max_tokens,
#             overlap_sentences=overlap_sentences
#         )

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple
import nltk
import re
from collections import namedtuple
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

DocumentChunk = namedtuple('DocumentChunk', ['content', 'embedding', 'metadata'])

class DocumentProcessor:
    def __init__(self, model_name: str = "multi-qa-mpnet-base-dot-v1"):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        # print(f"✅ Model loaded: {model_name}, Embedding Dimension: {self.embedding_dim}")

    def preprocess_text(self, text: str) -> str:
        """Clean text while preserving important phrases"""
        text = re.sub(r'\n\d+\s*\n', '\n', text)  # Remove page numbers
        text = re.sub(r'\s+', ' ', text)          # Normalize whitespace
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)  # Fix hyphenated words
        return text.strip()

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate normalized embeddings using model's native normalization"""
        if not text.strip():
            return np.zeros(self.embedding_dim)
        return self.model.encode(text, normalize_embeddings=True)

    def split_text_into_chunks(
        self,
        text: str,
        source_name: str = "",
        max_tokens: int = 150,  # Increased from 100
        overlap_sentences: int = 2  # Increased from 1
    ) -> List[DocumentChunk]:
        """Sentence-aware chunking with context overlap"""
        text = self.preprocess_text(text)
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_token_count = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = sentence.split()
            sentence_length = len(sentence_tokens)
            
            if current_token_count + sentence_length > max_tokens:
                if current_chunk:  # Finalize current chunk
                    chunks.append(" ".join(current_chunk))
                    # Carry over overlapping sentences
                    current_chunk = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
                    current_token_count = sum(len(s.split()) for s in current_chunk)
                
            current_chunk.append(sentence)
            current_token_count += sentence_length
        
        # Add remaining content
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        # Create DocumentChunk objects
        document_chunks = []
        for chunk_idx, chunk_text in enumerate(chunks):
            metadata = {
                "source": source_name,
                "chunk_index": chunk_idx,
                "sentences": len(sent_tokenize(chunk_text)),
                "token_count": len(chunk_text.split())
            }
            embedding = self.get_embedding(chunk_text)
            document_chunks.append(DocumentChunk(chunk_text, embedding, metadata))
        
        print(f"📄 Generated {len(document_chunks)} chunks with sentence-aware splitting")
        return document_chunks
    
    def find_relevant_chunks(
        self,
        query: str,
        chunks: List[DocumentChunk],
        top_k: int = 5
    ) -> List[Tuple[DocumentChunk, float]]:
        print(f"🔍 Searching here : {query}")
        """Hybrid search combining semantic similarity with exact matching"""
        query_embedding = self.get_embedding(query)
        similarities: List[Tuple[DocumentChunk, float]] = []

        key_terms = set(query.lower().split()) - {
            'a', 'the', 'and', 'or', 'in', 'on', 'at', 'is', 'are', 'be', 'to'
        }

        for chunk in chunks:
            # Skip bad embeddings
            if chunk.embedding.shape[0] != self.embedding_dim:
                print(f"⚠️ Dimension mismatch in chunk {chunk.metadata['chunk_index']}")
                continue

            # 1) Base semantic score
            similarity = float(np.dot(query_embedding, chunk.embedding))

            # 2) Substring boost
            if query.lower() in chunk.content.lower():
                similarity += 0.3

            # 3) Exact-sentence match boost + debug print
            sentences = sent_tokenize(chunk.content)
            if any(s.strip().lower() == query.strip().lower() for s in sentences):
                similarity += 1.0
                # Debug output:
                print(
                    f"🐞 Exact sentence matched in chunk {chunk.metadata['chunk_index']} "
                    f"(source={chunk.metadata['source']}):\n    \"{query}\""
                )
                # Or to drop into pdb:
                # import pdb; pdb.set_trace()

            # 4) Key-term bonus
            term_matches = sum(1 for term in key_terms if term in chunk.content.lower())
            similarity += 0.05 * term_matches

            similarities.append((chunk, similarity))

        # Return top-K
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def hybrid_search(
        self, 
        query: str, 
        chunks: List[DocumentChunk], 
        top_k: int = 5, 
        alpha: float = 0.7
    ) -> List[Tuple[DocumentChunk, float]]:
        """Combine semantic search with BM25 keyword search"""
        # Semantic search
        query_embedding = self.get_embedding(query)
        semantic_scores = []
        for chunk in chunks:
            similarity = np.dot(query_embedding, chunk.embedding)
            semantic_scores.append((chunk, similarity))
        
        # BM25 keyword search
        tokenized_chunks = [chunk.content.split() for chunk in chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        tokenized_query = query.split()
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # Normalize BM25 scores
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        normalized_bm25 = [score/max_bm25 for score in bm25_scores]
        
        # Combine scores
        final_scores = []
        for i, (chunk, semantic_score) in enumerate(semantic_scores):
            # Add exact match bonus
            # exact_match_bonus = 0.2 if query.lower() in chunk.content.lower() else 0
            exact_match_bonus = 0
            for sentence in sent_tokenize(chunk.content):
                if sentence.strip().lower() == query.strip().lower():
                    exact_match_bonus = 1.0  # Strong boost for exact sentence
                    break
            # Combine scores with weighting
            combined_score = (alpha * semantic_score) + ((1-alpha) * normalized_bm25[i]) + exact_match_bonus
            final_scores.append((chunk, combined_score))
        
        final_scores.sort(key=lambda x: x[1], reverse=True)
        return final_scores[:top_k]

    # def rerank_results(
    #     self,
    #     query: str,
    #     initial_results: List[Tuple[DocumentChunk, float]],
    #     top_k: int = 3
    # ) -> List[Tuple[DocumentChunk, float]]:
    #     """Simple reranking based on exact matches and semantic relevance"""
    #     reranked = []
        
    #     # Extract chunks from initial results
    #     initial_chunks = [chunk for chunk, _ in initial_results]
        
    #     for chunk in initial_chunks:
    #         base_score = 0.0
            
    #         # Check for exact sentence matches
    #         for sentence in sent_tokenize(chunk.content):
    #             sentence_lower = sentence.lower()
    #             query_lower = query.lower()
                
    #             # Exact sentence match is strongest signal
    #             if query_lower in sentence_lower:
    #                 base_score += 0.5
    #                 break
            
    #         # Check for key terms
    #         key_terms = set(query_lower.split()) - set(['a', 'the', 'and', 'or', 'in', 'on', 'at', 'is', 'are', 'be', 'to'])
    #         term_count = sum(1 for term in key_terms if term in chunk.content.lower())
    #         term_score = 0.1 * (term_count / max(1, len(key_terms)))
            
    #         # Combine scores
    #         final_score = base_score + term_score
            
    #         # Calculate separate semantic score for this specific chunk with query
    #         chunk_embedding = chunk.embedding
    #         query_embedding = self.get_embedding(query)
    #         semantic_score = np.dot(query_embedding, chunk_embedding)
            
    #         # Add semantic component to final score
    #         final_score += semantic_score
            
    #         reranked.append((chunk, final_score))
    #     print(f"🔄 Reranked {len(reranked)} results")
    #     # Sort by final score
    #     reranked.sort(key=lambda x: x[1], reverse=True)
    #     return reranked[:top_k]
    
    def rerank_results(self, query: str, initial_results: List[Tuple[DocumentChunk, float]], top_k: int = 3):
        pairs = [[query, chunk.content] for chunk, _ in initial_results]
        scores = self.reranker.predict(pairs)

        reranked = [
            (initial_results[i][0], float(scores[i]))
            for i in range(len(scores))
        ]
        reranked.sort(key=lambda x: x[1], reverse=True)
        print(f"🔄 CrossEncoder reranked {len(reranked)} results")
        return reranked[:top_k]

    
    def process_document(
        self,
        text: str,
        source_name: str,
        max_tokens: int = 150,  # Increased from 100
        overlap_sentences: int = 2  # Increased from 1
    ) -> List[DocumentChunk]:
        """End-to-end document processing pipeline"""
        return self.split_text_into_chunks(
            text=text,
            source_name=source_name,
            max_tokens=max_tokens,
            overlap_sentences=overlap_sentences
        )
    
    def semantic_chunking(
        self,
        text: str,
        source_name: str = "",
        max_tokens: int = 150
    ) -> List[DocumentChunk]:
        """Create chunks based on semantic similarity rather than just token count"""
        text = self.preprocess_text(text)
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        # Get embeddings for all sentences
        embeddings = self.model.encode(sentences, normalize_embeddings=True)
        
        for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
            tokens = sentence.split()
            
            # If adding this sentence exceeds max tokens, finalize chunk
            if current_tokens + len(tokens) > max_tokens and current_chunk:
                chunks.append(" ".join(current_chunk))
                
                # Find most similar sentence to start next chunk
                if i < len(sentences) - 1:
                    similarities = [np.dot(embedding, embeddings[j]) for j in range(i+1, min(i+5, len(sentences)))]
                    if similarities:
                        most_similar = similarities.index(max(similarities))
                        current_chunk = [sentences[i], sentences[i+1+most_similar]]
                        current_tokens = len(tokens) + len(sentences[i+1+most_similar].split())
                    else:
                        current_chunk = [sentence]
                        current_tokens = len(tokens)
                else:
                    current_chunk = [sentence]
                    current_tokens = len(tokens)
            else:
                current_chunk.append(sentence)
                current_tokens += len(tokens)
        
        # Add the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        # Create DocumentChunk objects
        document_chunks = []
        for chunk_idx, chunk_text in enumerate(chunks):
            metadata = {
                "source": source_name,
                "chunk_index": chunk_idx,
                "sentences": len(sent_tokenize(chunk_text)),
                "token_count": len(chunk_text.split())
            }
            embedding = self.get_embedding(chunk_text)
            document_chunks.append(DocumentChunk(chunk_text, embedding, metadata))
        
        print(f"📄 Generated {len(document_chunks)} chunks with semantic-aware splitting")
        return document_chunks