import torch

class BeamSearchDecoder:
    def __init__(self, tokenizer, penalty_weight=1.0, mass_tol=0.3,
                 repetition_penalty=1.2, length_penalty=1.0, no_repeat_ngram=2,
                 use_formula_head=True):
        """
        Beam search decoder with chemical plausibility checks and optional
        re-ranking using FormulaTransformerHead.

        Args:
            tokenizer: FormulaTokenizer instance
            penalty_weight: weight for soft plausibility penalties
            mass_tol: tolerance (Da) for precursor mass filtering
            repetition_penalty: >1.0 discourages repeating tokens
            length_penalty: >1.0 encourages longer sequences
            no_repeat_ngram: block repeated n-grams of this size
            use_formula_head: if True, re-rank beams with FormulaTransformerHead
        """
        self.tokenizer = tokenizer
        self.penalty_weight = penalty_weight
        self.mass_tol = mass_tol
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.no_repeat_ngram = no_repeat_ngram
        self.use_formula_head = use_formula_head

    def _has_repeat_ngram(self, seq_ids):
        """Check if the last n-gram has already appeared in the sequence."""
        if len(seq_ids) < self.no_repeat_ngram:
            return False
        ngram = tuple(seq_ids[-self.no_repeat_ngram:])
        for i in range(len(seq_ids) - self.no_repeat_ngram):
            if tuple(seq_ids[i:i+self.no_repeat_ngram]) == ngram:
                return True
        return False

    def decode(self, model, spectrum, device="cpu", beam_size=5, max_len=30, min_len=5,
               precursor_mass=None, formula_utils=None, return_k=5, element_order=None):
        """
        Run beam search decoding.

        Args:
            model: trained Spectrum2Formula model
            spectrum: input spectrum tensor (1D)
            device: torch.device
            beam_size: number of beams
            max_len: maximum sequence length
            min_len: minimum sequence length before allowing <eos>
            precursor_mass: optional precursor mass for filtering
            formula_utils: FormulaUtils class for plausibility checks
            return_k: number of top candidates to return
            element_order: list of element symbols (for formula head re-ranking)

        Returns:
            List of (formula_string, score) tuples
        """
        model.eval()
        with torch.no_grad():
            spectrum = spectrum.to(device).unsqueeze(0)
            memory = model.encoder(spectrum)

            beams = [(torch.tensor([[self.tokenizer.stoi["<sos>"]]], device=device), 0.0)]
            completed = []

            for step in range(max_len - 1):
                new_beams = []
                for seq, score in beams:
                    last = seq[0, -1].item()
                    if last == self.tokenizer.stoi["<eos>"]:
                        completed.append((seq, score))
                        continue

                    out = model.decoder(seq, memory)
                    probs = torch.log_softmax(out[0, -1], dim=-1)

                    # repetition penalty
                    for prev_token in set(seq[0].tolist()):
                        if probs[prev_token] < 0:
                            probs[prev_token] *= self.repetition_penalty
                        else:
                            probs[prev_token] /= self.repetition_penalty

                    topk = torch.topk(probs, beam_size)

                    for idx, logp in zip(topk.indices, topk.values):
                        if seq.size(1) < min_len and idx.item() == self.tokenizer.stoi["<eos>"]:
                            continue
                        new_seq = torch.cat([seq, idx.view(1, 1)], dim=1)
                        # n-gram blocking
                        if self._has_repeat_ngram(new_seq[0].tolist()):
                            continue
                        new_score = score + logp.item()
                        new_beams.append((new_seq, new_score))

                if not new_beams:
                    break
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

            completed.extend(beams)
            if not completed:
                return [("INVALID", 0.0)]

            # Decode sequences into formulas
            candidates = []
            for seq, score in completed:
                decoded = self.tokenizer.decode(seq.squeeze(0).tolist())
                fdict = formula_utils.parse_formula_dict(decoded)

                # Hard constraints
                total_valence = sum(formula_utils.VALENCE.get(e, 0) * c for e, c in fdict.items())
                if total_valence % 2 != 0:
                    continue
                if precursor_mass is not None:
                    pred_mass = formula_utils.compute_mass(fdict)
                    if abs(pred_mass - precursor_mass) > self.mass_tol:
                        continue

                # Soft penalties
                rules = formula_utils.check_all_rules(fdict)
                penalty = sum(1 for key in ["H/C", "N/C", "O/C"] if not rules.get(key, True))
                adjusted_score = score - self.penalty_weight * penalty

                # Length normalization
                length_norm = (len(seq.squeeze(0)) ** self.length_penalty)
                avg_log_prob = adjusted_score / max(1.0, length_norm)

                candidates.append((decoded, avg_log_prob))

            if not candidates:
                return [("INVALID", 0.0)]

            # Optional re-ranking with formula head
            if self.use_formula_head and hasattr(model, "formula_head") and model.formula_head is not None:
                formulas = [f for f, _ in candidates]
                counts = []
                for f in formulas:
                    fdict = formula_utils.parse_formula_dict(f)
                    counts.append([fdict.get(e, 0) for e in element_order])
                candidate_counts = torch.log1p(torch.tensor(counts, dtype=torch.float32, device=device)).unsqueeze(0)
                energies = model.score_formulas(spectrum, candidate_counts)  # (1, K)
                scores = energies.squeeze(0).tolist()
                reranked = sorted(zip(formulas, scores), key=lambda x: x[1], reverse=True)
                return reranked[:return_k]

            # Fallback: rank by avg_log_prob
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:return_k]
            # Convert avg_log_prob to exp for confidence
            return [(f, float(torch.exp(torch.tensor(s)))) for f, s in candidates]