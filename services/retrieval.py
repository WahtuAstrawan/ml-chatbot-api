import json
import logging
from functools import lru_cache
from typing import Set, Tuple, Any

import faiss
import numpy as np

from config.settings import get_settings
from services.embedding import get_embedding_service

logger = logging.getLogger(__name__)
settings = get_settings()


class RetrievalService:
    """Service for retrieving relevant context from the dataset using FAISS."""

    def __init__(self):
        """Initialize the retrieval service with FAISS index."""
        try:
            # Load dataset with pre-computed embeddings
            with open(settings.DATASET_PATH, 'r', encoding='utf-8') as f:
                self.data = json.load(f)

            if not self.data:
                raise ValueError("Dataset is empty")

            # Get corpus and embeddings from dataset
            self.corpus = [entry['text'] for entry in self.data]
            logger.info(f"Total bait yang ada: {len(self.corpus)}")

            # Extract pre-computed embeddings
            self.corpus_embeddings = np.array([entry['embedding'] for entry in self.data], dtype=np.float32)

            # Validate embeddings
            if self.corpus_embeddings.size == 0 or self.corpus_embeddings.shape[0] != len(self.corpus):
                raise ValueError("Invalid or missing embeddings in dataset")

            # Create FAISS index
            self.dimension = self.corpus_embeddings[0].shape[0]
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(self.corpus_embeddings)

            logger.info(f"FAISS index built with {len(self.corpus)} entries and dimension {self.dimension}")

        except FileNotFoundError:
            logger.error(f"Dataset file not found: {settings.DATASET_PATH}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format in dataset: {settings.DATASET_PATH}")
            raise
        except KeyError as e:
            logger.error(f"Missing 'embedding' or 'text' field in dataset: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error initializing retrieval service: {str(e)}")
            raise

    def retrieve(self, query: str, top_k: int = 3, context_size: int = 10)-> list[dict[str, Any]]:
        """
        Retrieve relevant context entries using FAISS.

        Args:
            query: Query string
            top_k: Number of top results to consider
            context_size: Number of surrounding baits to include

        Returns:
            retrieved_entries with is_top_k boolean attribute
        """
        try:
            # Encode query
            embedding_service = get_embedding_service()
            query_embedding = embedding_service.encode([query])

            # Search FAISS index
            distances, indices = self.index.search(np.array(query_embedding), top_k)

            # Store top-k indices as set of (sargah_number, bait) for quick lookup
            top_k_indices = indices[0][:top_k].tolist()
            top_k_baits = {(self.data[idx]['sargah_number'], self.data[idx]['bait']) for idx in top_k_indices}

            # Process results
            retrieved_entries = []
            seen_baits: Set[Tuple[int, int]] = set()

            for idx in indices[0]:
                sargah_num = self.data[idx]['sargah_number']
                bait_num = self.data[idx]['bait']

                # Get all baits in the same sargah
                same_sargah_entries = [e for e in self.data if e['sargah_number'] == sargah_num]
                same_sargah_entries.sort(key=lambda x: x['bait'])

                # Find index of selected bait in sargah
                selected_entry_idx = next(i for i, e in enumerate(same_sargah_entries) if e['bait'] == bait_num)
                total_baits = len(same_sargah_entries)

                # Get context_size baits before and after
                start_idx = max(0, selected_entry_idx - context_size)
                end_idx = min(total_baits, selected_entry_idx + context_size + 1)

                # Add baits from start_idx to end_idx, avoid duplicates
                for entry in same_sargah_entries[start_idx:end_idx]:
                    bait_id = (entry['sargah_number'], entry['bait'])
                    if bait_id not in seen_baits:
                        # Create a new entry with is_top_k flag
                        new_entry = entry.copy()  # Avoid modifying original data
                        new_entry['is_top_k'] = bait_id in top_k_baits
                        retrieved_entries.append(new_entry)
                        seen_baits.add(bait_id)

                # If fewer than 2*context_size + 1 baits, add more baits around
                current_count = sum(1 for e in retrieved_entries if e['sargah_number'] == sargah_num)
                expected_count = 2 * context_size + 1

                if current_count < expected_count:
                    remaining_needed = expected_count - current_count

                    # Try to add baits before start_idx, but not across sargahs
                    if start_idx > 0:
                        extra_start_idx = max(0, start_idx - remaining_needed)
                        for entry in same_sargah_entries[extra_start_idx:start_idx]:
                            bait_id = (entry['sargah_number'], entry['bait'])
                            if bait_id not in seen_baits:
                                new_entry = entry.copy()
                                new_entry['is_top_k'] = bait_id in top_k_baits
                                retrieved_entries.append(new_entry)
                                seen_baits.add(bait_id)
                                current_count += 1
                                if current_count >= expected_count:
                                    break

                    # If still not enough, add baits after end_idx
                    if current_count < expected_count:
                        extra_end_idx = min(total_baits, end_idx + (expected_count - current_count))
                        for entry in same_sargah_entries[end_idx:extra_end_idx]:
                            bait_id = (entry['sargah_number'], entry['bait'])
                            if bait_id not in seen_baits:
                                new_entry = entry.copy()
                                new_entry['is_top_k'] = bait_id in top_k_baits
                                retrieved_entries.append(new_entry)
                                seen_baits.add(bait_id)
                                current_count += 1
                                if current_count >= expected_count:
                                    break

            # Sort by sargah and bait number
            retrieved_entries.sort(key=lambda x: (x['sargah_number'], x['bait']))

            logger.debug(f"Retrieved {len(retrieved_entries)} entries for query: {query}")

            return retrieved_entries

        except Exception as e:
            logger.error(f"Error retrieving with FAISS: {str(e)}")
            raise


@lru_cache()
def get_retrieval_service() -> RetrievalService:
    """Get or create a singleton instance of RetrievalService."""
    return RetrievalService()