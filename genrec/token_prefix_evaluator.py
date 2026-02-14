from collections import defaultdict

class PrefixGramMemorizationEvaluator:
    """Evaluator for prefix-gram memorization.
    
    Analogue to FineGrainedEvaluator but operates on token prefixes.
    Checks for 'Contractivity' (forward reachability) in the prefix graph.
    
    Parameters:
        train_item_seqs: List of item sequences from training data
        tokenizer: Tokenizer with item2tokens mapping
        prefix_length: Number of prefix tokens to consider (e.g., 4)
        max_hop: Maximum hop to consider
    """
    
    def __init__(self, train_item_seqs, tokenizer, prefix_length=4, max_hop=4):
        self.tokenizer = tokenizer
        self.prefix_length = prefix_length
        self.max_hop = max_hop
        
        # rules[hop][prefix_u] -> set of prefix_v
        self.rules = {hop: defaultdict(set) for hop in range(1, max_hop + 1)}
        self._build_rules(train_item_seqs)
    
    def _item_tokens(self, item):
        """Return list of tokens for an item."""
        if item not in self.tokenizer.item2tokens:
            return []
        tokens = self.tokenizer._token_single_item(item)
        if isinstance(tokens, (tuple, list)):
            return list(tokens)
        return [tokens]
    
    def _extract_prefix(self, tokens):
        """Extract prefix of specified length from tokens.
        
        Returns:
            Tuple of first N tokens (where N = prefix_length)
        """
        if not tokens:
            return None
        if len(tokens) < self.prefix_length:
            return tuple(tokens)
        return tuple(tokens[:self.prefix_length])
    
    def _build_rules(self, train_item_seqs):
        """Build rules from training sequences (Inclusive Definition).
        
        If prefix(u) -> prefix(v) appears at distance D, 
        it is added to rules for all Hops H >= D.
        """
        for item_seq in train_item_seqs:
            n = len(item_seq)
            for i in range(1, n):
                target_item = item_seq[i]
                target_tokens = self._item_tokens(target_item)
                target_prefix = self._extract_prefix(target_tokens)
                
                if not target_prefix:
                    continue
                
                # Check previous items (Context)
                for ctx_length in range(1, self.max_hop + 1):
                    j = i - ctx_length
                    if j < 0:
                        continue
                    
                    context_item = item_seq[j]
                    context_tokens = self._item_tokens(context_item)
                    context_prefix = self._extract_prefix(context_tokens)
                    
                    if not context_prefix:
                        continue
                    
                    # Add edge to all hops >= distance (Inclusive Logic)
                    for hop in range(ctx_length, self.max_hop + 1):
                        self.rules[hop][context_prefix].add(target_prefix)

    def has_prefix_contractivity(self, u_prefix, v_prefix, hop):
        """Check if u_prefix -> v_prefix exists in training data at this hop."""
        if not u_prefix or not v_prefix:
            return False
        return v_prefix in self.rules[hop][u_prefix]
    
    def get_case_labels(self, item_seq, target_tokens=None):
        labels = set()
        if len(item_seq) < 2:
            return labels
        
        target_tokens = target_tokens or self._item_tokens(item_seq[-1])
        target_prefix = self._extract_prefix(target_tokens)
        
        if not target_prefix:
            if self.verbose: print(f"[Fail] Could not extract prefix for target.")
            return labels

        # Iterate Hops (1 to Max)
        for hop in range(1, self.max_hop + 1):
            
            # Reset flag for this specific hop level
            found_at_this_hop = False
            
            # Iterate Window (Distance 1 to Hop)
            for dist in range(1, hop + 1):
                # FIX: Check bounds here. 
                # If we don't have enough history for THIS distance, just skip this distance.
                # We do NOT skip the whole hop, because smaller distances might still work.
                if len(item_seq) - dist - 1 < 0:
                    continue 
                
                u_item = item_seq[-dist - 1]
                u_tokens = self._item_tokens(u_item)
                u_prefix = self._extract_prefix(u_tokens)
                
                if not u_prefix:
                    continue
                
                # Check reachability in Prefix Graph
                if self.has_prefix_contractivity(u_prefix, target_prefix, hop):
                    # Use your preferred naming convention
                    labels.add(f"memorization_{hop}")

                    found_at_this_hop = True
                    break # Break distance loop

            # If found at this hop, stop checking higher hops
            if found_at_this_hop:
                break
        
        if not labels:
            labels.add("unseen")
            
        return labels