from collections import defaultdict
import numpy as np


class FineGrainedEvaluator:
    """
    Helper class for fine-grained evaluation based on memorization and generalization patterns.
    Reuses logic from statistics.py and fine-grained-results.py.
    """
    
    def __init__(self, train_item_seqs, max_hop=4):
        """
        Initialize the fine-grained evaluator.
        
        Args:
            train_item_seqs: List of item sequences from training data
            max_hop: Maximum hop to consider (default: 4)
        """
        self.max_hop = max_hop
        self.logic2judger = {
            'substitutability': self.has_substitutability,
            'symmetry': self.has_symmetry,
            'transitivity': self.has_transitivity,
            '2nd-symmetry': self.has_2nd_symmetry,
        }
        self.rules = {}
        self.reverse_rules = {}
        self._build_rules(train_item_seqs)
    
    def _build_rules(self, train_item_seqs):
        """Build rules from training sequences."""
        self.rules, self.reverse_rules = {}, {}
        for hop in range(1, self.max_hop + 1):
            self.rules[hop] = defaultdict(set)
            self.reverse_rules[hop] = defaultdict(set)
        
        for item_seq in train_item_seqs:
            for i in range(1, len(item_seq)):
                v = item_seq[i]
                for ctx_length in range(1, self.max_hop + 1):
                    if i - ctx_length < 0:
                        continue
                    u = item_seq[i - ctx_length]
                    for hop in range(ctx_length, self.max_hop + 1):
                        self.rules[hop][u].add(v)
                        self.reverse_rules[hop][v].add(u)
    
    def has_memorization(self, u, v):
        """Check if u -> v exists in training data."""
        hop = 1
        return v in self.rules[hop][u]
    
    def has_substitutability(self, u, v, hop):
        """Check if u -> v exists in training data."""
        return v in self.rules[hop][u]
    
    def has_symmetry(self, u, v, hop):
        """Check if v -> u exists in training data."""
        return u in self.rules[hop][v]
    
    def has_transitivity(self, u, v, hop):
        """Check if u -> x -> v exists in training data."""
        return len(self.rules[hop][u].intersection(self.reverse_rules[hop][v])) > 0
    
    def has_2nd_symmetry(self, u, v, hop):
        """Check if second-order symmetry exists."""
        # v -> x -> u
        if self.rules[hop][v].intersection(self.reverse_rules[hop][u]):
            return True
        # v -> x <- u
        if self.rules[hop][v].intersection(self.rules[hop][u]):
            return True
        # v <- x -> u
        if self.reverse_rules[hop][v].intersection(self.reverse_rules[hop][u]):
            return True
        return False

    @property
    def ordered_keys(self):
        """Define the strict column order for the paper table."""
        keys = ["memorization", "generalization"]
        for logic in self.logic2judger.keys():
            for hop in range(1, self.max_hop + 1):
                # Skip substitutability_1 because it is equivalent to memorization
                if hop == 1 and logic == 'substitutability':
                    continue
                label = f"{logic}_{hop}"
                keys.append(label)
        keys.append("uncategorized")
        return keys
    
    def get_case_labels(self, item_seq):
        """
        Get labels for a single test case.
        
        Args:
            item_seq: Item sequence to label
            
        Returns:
            Set of labels (e.g., {'memorization_2', 'uncategorized_3'})
        """
        labels = set()
        if len(item_seq) < 2:
            return labels
        
        v = item_seq[-1]
        
        # check memorization
        if self.has_memorization(item_seq[-2], v):
            labels.add('memorization')
            return labels

        # check generalization patterns
        for logic, judger in self.logic2judger.items():
            # substitutability-1 is memorization, so we start from hop 2
            min_hop = 2 if logic == 'substitutability' else 1
            for hop in range(min_hop, self.max_hop + 1):
                if len(item_seq) - hop - 1 < 0:
                    continue
                found_at_this_hop = False
                # consider all context items within k-hop window
                for dist in range(1, hop + 1):
                    u = item_seq[-dist - 1]
                    if judger(u, v, hop):
                        labels.add(f"{logic}_{hop}")
                        found_at_this_hop = True
                        break
                # if found at this hop, no need to check higher hops
                if found_at_this_hop:
                    break
        # add master label
        if not labels:
            labels.add(f"uncategorized")
        else:
            labels.add("generalization")
        
        return labels
    
    def compute_pattern_statistics(self, split_item_seqs):
        """
        Compute pattern statistics (ratios) for a given split.
        Reuses get_case_labels to ensure logic consistency.
        """
        n_instances = len(split_item_seqs)
        assert n_instances > 0, "No instances to compute statistics"
            
        counts = defaultdict(float)
        for item_seq in split_item_seqs:
            labels = self.get_case_labels(item_seq)
            for label in labels:
                counts[label] += 1
        
        ratios = {}
        for logic in self.logic2judger.keys():
            for hop in range(1, self.max_hop + 1):
                if logic == 'substitutability' and hop == 1:
                    label = "memorization"
                else:
                    label = f"{logic}_{hop}"
                ratios[label] = counts[label] / n_instances
        
        ratios['uncategorized'] = counts['uncategorized'] / n_instances
        return ratios
