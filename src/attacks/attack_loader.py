import json
import os
import random
from typing import List, Dict, Optional
from .external_loader import ExternalDatasetLoader

class AttackLoader:
    def __init__(self, data_path: Optional[str] = None):
        if data_path is None:
            # Default path: ../../data/attacks/static_attacks.json
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            data_path = os.path.join(base_dir, 'data', 'attacks', 'static_attacks.json')
            
        self.data_path = data_path
        self.metadata = {}
        self.attacks = []
        
        self.load_data()
        self.external_loader = ExternalDatasetLoader()
        
    def load_data(self):
        """Loads the static attack dataset from JSON."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Attack dataset not found at {self.data_path}")
            
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.metadata = data.get("metadata", {})
            self.attacks = data.get("attacks", [])
            
    def get_all_attacks(self) -> List[Dict]:
        """Returns all attacks in the dataset."""
        return self.attacks
        
    def get_attacks_by_category(self, category: str) -> List[Dict]:
        """Filters attacks by primary category."""
        return [a for a in self.attacks if a["category"] == category]
        
    def get_attacks_by_subcategory(self, subcategory: str) -> List[Dict]:
        """Filters attacks by subcategory."""
        return [a for a in self.attacks if a["subcategory"] == subcategory]
        
    def get_attacks_by_difficulty(self, difficulty: str) -> List[Dict]:
        """Filters attacks by difficulty (easy, medium, hard)."""
        return [a for a in self.attacks if a.get("difficulty") == difficulty]
        
    def sample_attacks(self, n: int, category: Optional[str] = None) -> List[Dict]:
        """Returns a random sample of n attacks, optionally filtered by category."""
        pool = self.get_attacks_by_category(category) if category else self.attacks
        if n >= len(pool):
            return pool
        return random.sample(pool, n)

    def get_attack_by_id(self, attack_id: str) -> Optional[Dict]:
        """Returns a specific attack by its ID."""
        for a in self.attacks:
            if a["attack_id"] == attack_id:
                return a
        return None

    def load_external_dataset(self, dataset_name: str = "bipia", num_samples: int = 100) -> List[Dict]:
        """
        Loads an external dataset and dynamically adds it to the pool of attacks.
        Currently supports 'bipia' (using deepset/prompt-injections as proxy).
        """
        if dataset_name.lower() == "bipia":
            external_attacks = self.external_loader.load_bipia(num_samples=num_samples)
            # Add to our internal list if not already present
            existing_ids = {a["attack_id"] for a in self.attacks}
            new_attacks = [a for a in external_attacks if a["attack_id"] not in existing_ids]
            self.attacks.extend(new_attacks)
            print(f"Loaded {len(new_attacks)} new external attacks from {dataset_name}.")
            return external_attacks
        else:
            raise ValueError(f"Unknown external dataset: {dataset_name}")


class BenignQueryLoader:
    """Loads and queries the benign test query dataset for FPR testing."""

    def __init__(self, data_path: Optional[str] = None):
        if data_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            data_path = os.path.join(base_dir, 'data', 'benign', 'benign_queries.json')

        self.data_path = data_path
        self.metadata = {}
        self.queries = []

        self.load_data()

    def load_data(self):
        """Loads the benign query dataset from JSON."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Benign query dataset not found at {self.data_path}")

        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.metadata = data.get("metadata", {})
            self.queries = data.get("queries", [])

    def get_all_queries(self) -> List[Dict]:
        """Returns all benign queries."""
        return self.queries

    def get_queries_by_topic(self, topic: str) -> List[Dict]:
        """Filters queries by topic (ai_ml, computer_security, general_technology, edge_case)."""
        return [q for q in self.queries if q["topic"] == topic]

    def get_queries_by_type(self, query_type: str) -> List[Dict]:
        """Filters queries by type (factual, explanatory, comparative, how_to, opinion, troubleshooting)."""
        return [q for q in self.queries if q["query_type"] == query_type]

    def get_edge_cases(self) -> List[Dict]:
        """Returns only edge-case queries (those with injection-like keywords)."""
        return self.get_queries_by_topic("edge_case")

    def sample_queries(self, n: int, topic: Optional[str] = None) -> List[Dict]:
        """Returns a random sample of n queries, optionally filtered by topic."""
        pool = self.get_queries_by_topic(topic) if topic else self.queries
        if n >= len(pool):
            return pool
        return random.sample(pool, n)

    def get_query_by_id(self, query_id: str) -> Optional[Dict]:
        """Returns a specific query by its ID."""
        for q in self.queries:
            if q["query_id"] == query_id:
                return q
        return None


if __name__ == "__main__":
    # Test AttackLoader
    attack_loader = AttackLoader()
    print(f"Loaded {len(attack_loader.get_all_attacks())} total attacks.")
    for cat in attack_loader.metadata.get('distribution', {}).keys():
        count = len(attack_loader.get_attacks_by_category(cat))
        print(f"  - {cat}: {count} attacks")

    print()

    # Test BenignQueryLoader
    benign_loader = BenignQueryLoader()
    print(f"Loaded {len(benign_loader.get_all_queries())} total benign queries.")
    for topic in benign_loader.metadata.get('distribution', {}).keys():
        count = len(benign_loader.get_queries_by_topic(topic))
        print(f"  - {topic}: {count} queries")
    print(f"  - edge cases: {len(benign_loader.get_edge_cases())} queries")
