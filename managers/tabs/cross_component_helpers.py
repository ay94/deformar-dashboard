from typing import Dict, Iterator, Iterable, Tuple, List, Optional
import pandas as pd
import numpy as np
from scipy.stats import rankdata
from seqeval.metrics.sequence_labeling import get_entities
from seqeval.scheme import Entities, auto_detect
from collections import Counter, defaultdict


_SPLIT_LABEL    = {"train": "Train", "test": "Test", "validation": "Validation", "dev": "Validation"}
_DATASET_LABEL  = {"ANERCorp_CamelLab": "ANERCorp", "conll2003": "CoNLL-2003"}


# pretty/variant names -> raw corpus keys (extend as needed)
_DATASET_MAP    = {
    "ANERCorp": "ANERCorp_CamelLab",
    "CoNLL-2003": "conll2003",
    "ANERCorp_CamelLab_arabertv02": "ANERCorp_CamelLab",
    "conll2003_bert": "conll2003",
}

LANGUAGE_MAP = {
    "ANERCorp_CamelLab": "Arabic",
    "conll2003": "English",
    "ANERCorp_CamelLab_arabertv02": "Arabic",
    "conll2003_bert": "English",
}


_TAG_NORMALIZE  = {"B-PERS": "B-PER", "I-PERS": "I-PER"}

_DEFAULT_TAGS = ['B-LOC','I-LOC','B-PER','I-PER','B-ORG','I-ORG','B-MISC','I-MISC']

BASE_DATASETS = {"ANERCorp_CamelLab", "conll2003"}


_DEFAULT_ENTITY_SPANS = ["LOC", "PER", "ORG", "MISC"]

_SPAN_NORMALIZE = {"PERS": "PER"}  # unify ANER's PERS→PER


EXCLUDE_ROWS = {"micro", "macro", "weighted", "accuracy/micro"}
ROW_ORDER = ["Precision", "Recall", "F1-score"]

class BaseDatasetHelper:
    """Only cross-cutting utilities. No analysis-specific logic here."""

    def __init__(self, corpora: Dict):
        self.corpora = corpora

    # ---------- variant → dataset keys ----------
    def resolve_language_keys(self, variant: str) -> List[str]:
        """Return raw corpus keys for a given variant or 'combined'."""
        if variant == "combined":
            # choose the ones you want to include in 'combined'
            # return [k for k in self.corpora.keys() if k in ("ANERCorp_CamelLab", "conll2003")]
            return [k for k in self.corpora.keys() if k in BASE_DATASETS]
        # pretty/alias → raw
        if variant in _DATASET_MAP:
            return [_DATASET_MAP[variant]]
        # # assume it's already a raw key
        return [variant]

    # ---------- labels ----------
    def lang_label(self, ds_key: str) -> str:
        return LANGUAGE_MAP.get(ds_key, ds_key)

    def split_label(self, split_key: str) -> str:
        return _SPLIT_LABEL.get(split_key.lower(), split_key.title())

    # ---------- iterate splits ----------
    def iter_splits(self, ds_key: str) -> Iterator[Tuple[str, str, pd.DataFrame]]:
        """
        Yields (split_key, split_label, df) for a dataset key.
        - Prefers train/test if present
        - Drops validation for conll2003 (customize rules here)
        """
        splits = self.corpora[ds_key]["splits"]
        order = [k for k in ("train", "test") if k in splits] or list(splits.keys())
        if ds_key == "conll2003":
            order = [k for k in order if k.lower() != "validation"]
        for sk in order:
            yield sk, self.split_label(sk), splits[sk]

    # ---------- tag normalization (optional) ----------
    def normalize_tag_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure a 'Tag' column exists and normalize label variants.
        Accepts 'True Labels' or 'Tag' as source.
        """
        tag_col = "True Labels" if "True Labels" in df.columns else ("Tag" if "Tag" in df.columns else None)
        if tag_col is None:
            raise ValueError("Expected a tag column: 'True Labels' or 'Tag'.")
        out = df.copy()
        out[tag_col] = out[tag_col].replace(_TAG_NORMALIZE)
        return out.rename(columns={tag_col: "Tag"})
    
    @staticmethod
    def _normalize_tag(tag: str) -> str:
        return _TAG_NORMALIZE.get(tag, tag)




class BaseDashDataProcessor:
    def __init__(self, dash_data: Dict[str, "DashboardData"]):
        self.dash_data = dash_data  # raw objects with .train_data / .analysis_data
        self.corpora = {}           # filled by build_corpora()

    # labels (optional pretty names)
    # def ds_label(self, variant_key: str) -> str:
    #     """
    #     Example:
    #       variant_key='ANERCorp_CamelLab_arabertv02'
    #       DATA_MAP[variant_key] -> 'ANERCorp_CamelLab'
    #       _DATASET_LABEL['ANERCorp_CamelLab'] -> 'ANERCorp'
    #     Falls back gracefully if mappings are missing.
    #     """
    #     ds_key = _DATASET_MAP.get(variant_key, variant_key)
    #     return _DATASET_LABEL.get(ds_key, ds_key)
    def ds_label(self, variant_key: str) -> str:
        return LANGUAGE_MAP.get(variant_key, variant_key)


    def normalise_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df['Labels'] != -100].copy()
        if 'True Labels' in df.columns:
            df['True Labels'] = df['True Labels'].replace({'B-PERS': 'B-PER', 'I-PERS': 'I-PER'})
        return df
    
    @staticmethod
    def normalize_spans(df: pd.DataFrame, col: str = "Tag") -> pd.DataFrame:
        """
        Normalize BIO span tags (e.g., PERS -> PER).
        Operates in-place on a copy of the DataFrame.
        """
        out = df.copy()
        if col in out.columns:
            out[col] = out[col].replace(_SPAN_NORMALIZE)
        return out

    def _resolve_keys_for_variant(self, variant: str) -> list[str]:
        """Map a variant to concrete keys in dash_data."""
        
        if variant == "combined":
            return list(self.dash_data.keys())
        return [variant]

    def build_corpora(self, variant: str) -> dict:
        """
        selected_variant:
          - '<variant_name>' e.g. 'ANERCorp_CamelLab_arabertv02'
          - 'combined'       -> include all variants present in dash_data

        Populate self.corpora for the requested variant.
        Output shape:
          self.corpora[<Dataset Label>]['splits'] = {'Train': df, 'Test': df}
        """
        self.corpora = {}
        variant_keys = self._resolve_keys_for_variant(variant)

        for variant_key in variant_keys:
            ds_lbl = self.ds_label(variant_key)

            # ---- get & normalize splits ----
            train_df = self.normalise_data(self.dash_data[variant_key].train_data)
            test_df  = self.normalise_data(self.dash_data[variant_key].analysis_data)

            # ensure expected columns exist
            if "Core Tokens" not in train_df.columns and "core_tokens" in train_df.columns:
                train_df = train_df.rename(columns={"core_tokens": "Core Tokens"})
            if "Core Tokens" not in test_df.columns and "core_tokens" in test_df.columns:
                test_df  = test_df.rename(columns={"core_tokens": "Core Tokens"})

            self.corpora[ds_lbl] = {
                "splits": {
                    "Train": train_df,
                    "Test":  test_df,
                }
            }

        return self.corpora
    
    def iter_splits(self, ds_key: str, only_test: bool = False) -> Iterator[Tuple[str, str, pd.DataFrame]]:
        """
        Yield dataset splits.
        Set only_test=True to restrict to Test split (ignore Train).
        """
        splits = ("Test",) if only_test else ("Train", "Test")
        for sk in splits:
            if sk in self.corpora[ds_key]["splits"]:
                yield sk, sk, self.corpora[ds_key]["splits"][sk]
    

    def process_entity_confusion(self, entity_confusion: dict, o_error: str):
        """
        Processes the entity confusion matrix into high-level error categories
        and a separate DataFrame for entity vs exclusion errors.

        Parameters
        ----------
        entity_confusion : dict
            A dictionary of confusion components (raw outputs).
        o_error : str
            Label to rename the 'O' column (e.g. 'Inclusion' or 'Exclusion').

        Returns
        -------
        renamed_df : pd.DataFrame
            High-level error categories:
            - Entity
            - Boundary
            - Entity and Boundary
            - O Errors (renamed from 'O')
        entity_errors : pd.DataFrame
            Entity-only vs O-only error breakdown.
        """

        # Step 1: Build DataFrame
        df = pd.DataFrame(entity_confusion).fillna(0).astype(int).T

        # Step 2: Collapse into high-level categories
        errors = df.copy()
        if "O" in errors.columns:
            errors[o_error] = errors.pop("O")  # Rename O → Inclusion/Exclusion/O Errors

        errors["Entity"] = errors.drop(
            columns=["Boundary", "Entity and Boundary", o_error], errors="ignore"
        ).sum(axis=1)

        # keep only the four categories we care about
        errors = errors[["Entity", "Boundary", "Entity and Boundary", o_error]]

        # Step 3: Make a simplified df (entity-only vs O-only)
        entity_errors = df.drop(
            columns=["Boundary", "Entity and Boundary", "O"], errors="ignore"
        )

        return errors, entity_errors



class DatasetStatsHelper(BaseDatasetHelper):
    @staticmethod
    def _iter_tokens_and_tags(sentences):
        for s in sentences:
            for w, t in zip(s["words"], s["tags"]):
                yield w, t

    def _compute_split_counts(self, split_sentences):
        # per-split token + NE counts (and per-split uniques)
        tokens = []
        ne_tokens = []
        ne_unique = set()
        for w, t in self._iter_tokens_and_tags(split_sentences):
            tokens.append(w)
            if t != "O":
                ne_tokens.append(w)
                ne_unique.add(w)
        total = len(tokens)
        unique_words = len(set(tokens))
        twr  = (unique_words / total) if total else 0.0          # ← TWR
        ne_words = len(ne_tokens)
        unique_ne_words = len(ne_unique)
        ne_prop = (ne_words / total) if total else 0.0
        ne_type_prop = (unique_ne_words / unique_words) if ne_words else 0.0
        tewr = (unique_ne_words / ne_words) if ne_words else 0.0                       # ← TEWR
        return {
            "Total Words": total,
            "NE Words": ne_words,
            "NE Proportion": ne_prop,
            "TWR": twr,                       # Type–Word Ratio (lexical diversity)
            "Unique Words": unique_words,
            "Unique NE Words": unique_ne_words,
            "NE Type Proportion": ne_type_prop,
            "TEWR": tewr,                     # NE Type–Word Ratio (NE diversity)
        }

    def generate_df(self, selected_variant: str, include_total: bool = True) -> pd.DataFrame:
        """
        Rows (index):
          - Total Words, Unique Words, NE Words, Unique NE Words, NE Proportion
        Cols:
          - MultiIndex (Language, Split[Train/Test/(Total)])
        """
        metrics = ["Total Words", "NE Words", "NE Proportion", "TWR", "Unique Words", "Unique NE Words", "NE Type Proportion", "TEWR"]
        cols, data = [], {m: [] for m in metrics}

        for ds_key in self.resolve_language_keys(selected_variant):
            ds_lbl = self.lang_label(ds_key)

            # We'll also accumulate across all splits to recompute 'Total' uniques properly
            all_tokens = []
            all_ne_tokens = []
            all_ne_unique = set()

            # per-split
            for sk, sk_lbl, split_sentences in self.iter_splits(ds_key):
                stats = self._compute_split_counts(split_sentences)
                cols.append((ds_lbl, sk_lbl))
                for m in metrics:
                    data[m].append(stats[m])

                # accumulate for overall recomputation
                for s in split_sentences:
                    all_tokens.extend(s["words"])
                    for w, t in zip(s["words"], s["tags"]):
                        if t != "O":
                            all_ne_tokens.append(w)
                            all_ne_unique.add(w)

            if include_total:
                total_tokens = len(all_tokens)
                unique_words = len(set(all_tokens))
                twr  = (unique_words / total_tokens) if total_tokens else 0.0          # ← TWR
                ne_words = len(all_ne_tokens)
                unique_ne_words = len(all_ne_unique)
                ne_prop = (ne_words / total_tokens) if total_tokens else 0.0
                ne_type_prop = (unique_ne_words / unique_words) if ne_words else 0.0
                tewr = (unique_ne_words / ne_words) if ne_words else 0.0                       # ← TEWR

                cols.append((ds_lbl, "Total"))
                totals_row = {
                    "Total Words": total_tokens,
                    "NE Words": ne_words,
                    "NE Proportion": ne_prop,
                    "TWR": twr,                       # Type–Word Ratio (lexical diversity)
                    "Unique Words": unique_words,
                    "Unique NE Words": unique_ne_words,
                    "NE Type Proportion": ne_type_prop,
                    "TEWR": tewr,                     # NE Type–Word Ratio (NE diversity)
                }
                for m in metrics:
                    data[m].append(totals_row[m])

        df = pd.DataFrame(data, index=pd.RangeIndex(len(cols))).T
        df.columns = pd.MultiIndex.from_tuples(cols, names=["Language", "Split"])
        return df

    

class EntityTagDistribution(BaseDatasetHelper):
    def generate_df(self, variant: str, tag_set: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Build a DataFrame with per-tag statistics.

        Columns:
            Split                : Split label (e.g. Train, Test).
            Tag                  : Entity tag (B-LOC, I-LOC, etc.).
            Tag Words            : Raw NE token count for this tag.
            Tag Types            : Unique word *forms* that appeared with this tag.
            TWR                  : Type–Word Ratio = Tag Types / Tag Words.
            Words Proportion     : Share of NE tokens in this tag
                                (denominator = all NE tokens in the split).
            Type Proportion      : Share of unique NE word *forms* in this tag
                                (denominator = union of all NE forms in the split).
            Tag Type Proportion  : Share of Tag Types in this tag
                                (denominator = sum of Tag Types across all tags in this split;
                                always sums to 1 per split).
            Dataset              : Dataset label (e.g. ANERCorp, CoNLL-2003).
        """
        use_tags = tag_set or _DEFAULT_TAGS
        rows = []
        for ds_key in self.resolve_language_keys(variant):
            ds_lbl = self.lang_label(ds_key)
            for _, sk_lbl, split_sentences in self.iter_splits(ds_key):
                use_tags = tag_set or None  # if None, _tag_stats_for_split derives locally
                # split_df = self._tag_stats_for_split(sk_lbl, split_sentences, use_tags)
                split_df = self._tag_stats_for_split(
                    split_name=sk_lbl,
                    split_sentences=split_sentences,
                    tag_set=use_tags
                )
                split_df["Language"] = ds_lbl
                # rows.append(split_df)
            
                # --- add Tag Type Proportion directly ---
                total_tag_types = split_df["Tag Types"].sum()
                if total_tag_types > 0:
                    split_df["Tag Type Proportion"] = split_df["Tag Types"] / total_tag_types
                else:
                    split_df["Tag Type Proportion"] = 0.0

                rows.append(split_df)
                

        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
            columns=[
                "Split", "Tag", 
                "Tag Words", "Tag Types", "TWR",                  
                "Words Proportion", "Type Proportion",      
                "Tag Type Proportion",   
                "Language"
                ]
        )
    
    def add_tag_type_proportion(df: pd.DataFrame) -> pd.DataFrame:
        totals = (
            df.groupby(["Language", "Split"], as_index=False)["Tag Types"]
            .sum()
            .rename(columns={"Tag Types": "Total Tag Types"})
        )
        out = df.merge(totals, on=["Language", "Split"], how="left")
        out["Tag Type Proportion"] = (out["Tag Types"] / out["Total Tag Types"]).fillna(0.0)
        return out

    def _tag_stats_for_split(
        self,
        split_name: str,
        split_sentences: List[Dict],
        *,
        tag_set: Optional[List[str]] = None,
        round_to: int = 2,
    ) -> pd.DataFrame:
        """
        If tag_set is None → discover tags just from this split.
        Otherwise → use the provided tag_set, filling in zeros for missing tags.
        """
        # derive tag_set if not passed
        if tag_set is None:
            tags = {self._normalize_tag(t) for sent in split_sentences for t in sent["tags"] if t != "O"}
            tag_set = sorted(tags)

        tag_counts = {t: 0 for t in tag_set}
        unique_tag_words = {t: set() for t in tag_set}

        total_ne_tokens = 0
        total_ne_unique_words = set()

        for sent in split_sentences:
            for w, t in zip(sent["words"], sent["tags"]):
                t = self._normalize_tag(t)
                if t != "O":
                    total_ne_tokens += 1
                    total_ne_unique_words.add(w)
                if t in tag_counts:
                    tag_counts[t] += 1
                    unique_tag_words[t].add(w)

        rows = []
        for t in tag_set:
            count = tag_counts[t]
            types = len(unique_tag_words[t])
            twr = (types / count) if count else 0.0
            token_prop = (count / total_ne_tokens) if total_ne_tokens else 0.0
            type_prop  = (types / len(total_ne_unique_words)) if total_ne_unique_words else 0.0
            rows.append({
                "Split": split_name,
                "Tag": t,
                "Tag Words": count,
                "Tag Types": types,
                "TWR": round(twr, round_to),
                "Words Proportion": round(token_prop, round_to),
                "Type Proportion": round(type_prop, round_to),
            })
        return pd.DataFrame(rows)


class EntitySpanDistribution(BaseDatasetHelper):
    """
    Builds span-level entity distributions.
    Returns columns:
      Dataset, Split, Scheme, Entity, Span Count, Span Proportion
    """

    def generate_df(
        self,
        variant: str,
        *,
        schemes: List[str] = ("IOB1", "IOB2"),
        splits: Optional[List[str]] = None,                # None → use iter_splits order
        entity_set: Optional[List[str]] = None,            # None → _DEFAULT_ENTITY_SET
        round_to: int = 2,
    ) -> pd.DataFrame:

        entity_span_set = entity_set or _DEFAULT_ENTITY_SPANS
        rows: List[Dict] = []

        for ds_key in self.resolve_language_keys(variant):
            ds_lbl = self.lang_label(ds_key)

            split_iter = list(self.iter_splits(ds_key))
            if splits:
                split_iter = [(sk, sk_lbl, sents) for (sk, sk_lbl, sents) in split_iter if sk in splits]

            for sk, sk_lbl, split_sentences in split_iter:
                # gold tags for this split
                y_true = [sen["tags"] for sen in split_sentences]

                for version in schemes:
                    # --- keep YOUR original extraction paths exactly ---
                    if version == "IOB1":
                        true_entities = get_entities(y_true)
                        # true_entities: list of tuples; count by entity[0]
                    else:
                        scheme = auto_detect(y_true, False)
                        entities = Entities(y_true, scheme, False)
                        # flatten per-sentence entities; to_tuple() → (sent_id, type, start, end)
                        true_entities = [entity.to_tuple()[1:] for sen in entities.entities for entity in sen]

                    # normalise labels and count only what we care about
                    raw_counts = Counter(
                        _SPAN_NORMALIZE.get(ent[0], ent[0]) for ent in true_entities
                    )

                    # filter to requested entity_set (keeps zero rows consistent)
                    counts_in_set = {e: raw_counts.get(e, 0) for e in entity_span_set}
                    total = sum(counts_in_set.values())

                    for entity_span in entity_span_set:
                        count = counts_in_set[entity_span]
                        prop = (count / total) if total else 0.0
                        rows.append({
                            "Language": ds_lbl,            # e.g., ANERCorp / CoNLL-2003
                            "Split": sk_lbl,                  # keep lower-case like your other tables
                            "Scheme": version,            # IOB1 / IOB2
                            "Entity": entity_span,        # PER / LOC / ORG / MISC
                            "Span Count": count,
                            "Span Proportion": round(prop, round_to),
                        })

        return pd.DataFrame(rows, columns=[
            "Language", "Split", "Scheme", "Entity", "Span Count", "Span Proportion"
        ])


class EntitySpanComplexity(BaseDatasetHelper):
    """
    Produce one row per entity span:
      Columns: Language, Split, Scheme, Entity, Span Length
    Use this directly with a faceted BOX PLOT.
    """

    def generate_df(
        self,
        variant: str,
        *,
        schemes=("IOB1","IOB2"),
        splits=None,                    # e.g. ["train","test"] or None → iter_splits order
        strict_iob2: bool = True,       # how you compute length for IOB2: end-start(+1)
    ) -> pd.DataFrame:
        rows = []

        for ds_key in self.resolve_language_keys(variant):
            ds_lbl = self.lang_label(ds_key)

            split_iter = list(self.iter_splits(ds_key))
            if splits:
                split_iter = [(sk, sk_lbl, sents) for (sk, sk_lbl, sents) in split_iter if sk in splits]

            for sk, sk_lbl, split_sentences in split_iter:
                y_true = [sen["tags"] for sen in split_sentences]

                for scheme_name in schemes:
                    if scheme_name == "IOB1":
                        # Your original path
                        # get_entities returns list of (type, start, end) across corpus
                        true_entities = get_entities(y_true)
                        # inclusive length for IOB1 (matches your previous strict=False):
                        for ent_type, start, end in true_entities:
                            ent_type = _SPAN_NORMALIZE.get(ent_type, ent_type)
                            length = (end - start + 1)
                            rows.append({
                                "Language": ds_lbl,
                                "Split": sk_lbl,                 # keep lower-case for consistency
                                "Scheme": scheme_name,
                                "Entity": ent_type,
                                "Span Length": int(length),
                            })
                    else:
                        # Your original path for IOB2
                        scheme = auto_detect(y_true, False)
                        ent_obj = Entities(y_true, scheme, False)
                        # per sentence entities; to_tuple() = (sent_id, type, start, end)
                        for sen in ent_obj.entities:
                            for e in sen:
                                _, ent_type, start, end = e.to_tuple()
                                ent_type = _SPAN_NORMALIZE.get(ent_type, ent_type)
                                if strict_iob2:
                                    # strict=True in your earlier class meant end-start (exclusive end)
                                    # but your text says: “mean span lengths … average number of tokens”
                                    # For tokens, inclusive is standard, so mirror IOB1:
                                    length = (end - start + 1)
                                else:
                                    length = (end - start)
                                rows.append({
                                    "Language": ds_lbl,
                                    "Split": sk_lbl,
                                    "Scheme": scheme_name,
                                    "Entity": ent_type,
                                    "Span Length": int(length),
                                })

        return pd.DataFrame(rows, columns=["Language", "Split", "Scheme", "Entity", "Span Length"])


class WordTypeFrequencyDistribution(BaseDatasetHelper):
    """
    For each (Language, Split, Tag), compute word-type frequency stats:
      - Mean Frequency
      - Median Frequency
      - Standard Deviation
    Word-type frequencies are counts of unique word forms appearing with that tag
    within the split (e.g., how many times each distinct name tagged B-PER appears).
    """

    def __init__(self, corpora: Dict, tags_of_interest: Optional[List[str]] = None):
        super().__init__(corpora)
        self.tags = tags_of_interest or _DEFAULT_TAGS

    def generate_df(self, variant: str, round_to: int = 2, splits: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Returns a tidy DataFrame with columns:
          Language, Split, Tag, Mean Frequency, Median Frequency, Standard Deviation
        """
        rows: List[dict] = []

        for ds_key in self.resolve_language_keys(variant):
            ds_lbl = self.lang_label(ds_key)

            # walk the available splits with your built-in iterator
            for split_key, split_lbl, split_sentences in self.iter_splits(ds_key):
                # if the caller restricts splits, skip others
                if splits and split_key not in splits:
                    continue

                # count word frequencies per tag for THIS split
                per_tag_counters = {tag: Counter() for tag in self.tags}
                for sent in split_sentences:
                    for w, t in zip(sent["words"], sent["tags"]):
                        t = self._normalize_tag(t)  # PERS→PER etc.
                        if t in per_tag_counters:
                            per_tag_counters[t][w] += 1

                # turn per-tag counters into stats
                for tag, counter in per_tag_counters.items():
                    freqs = np.fromiter(counter.values(), dtype=np.int64) if counter else np.array([], dtype=np.int64)
                    if freqs.size:
                        mean_f = float(np.mean(freqs))
                        med_f  = float(np.median(freqs))
                        std_f  = float(np.std(freqs, ddof=0))
                    else:
                        mean_f = med_f = std_f = 0.0

                    rows.append({
                        "Language": ds_lbl,
                        "Split": split_lbl,
                        "Tag": tag,
                        "Mean Frequency": round(mean_f, round_to),
                        "Median Frequency": round(med_f, round_to),
                        "Standard Deviation": round(std_f, round_to),
                    })

        return pd.DataFrame(rows)
    



class DatasetOOVRate(BaseDatasetHelper):
    """Rows = datasets, Cols = metrics (matches your example)."""

    def generate_df(self, selected_variant: str, round_to: int = 4) -> pd.DataFrame:
        rows = []
        seen_labels = set()

        for ds_key in self.resolve_language_keys(selected_variant):
            ds_lbl = self.lang_label(ds_key)
            if ds_lbl in seen_labels:
                continue
            seen_labels.add(ds_lbl)

            splits = self.corpora[ds_key]["splits"]
            train = splits.get("train")
            test  = splits.get("test")

            if not train or not test:
                rows.append({
                    "Language": ds_lbl,
                    "OOV Words Count": 0.0,
                    "Total Unique Words in Test": 0.0,
                    "OOV Rate": 0.0,
                })
                continue

            train_words = {w for s in train for w in s["words"]}
            test_words  = {w for s in test  for w in s["words"]}
            oov_words   = test_words - train_words
            oov_rate    = (len(oov_words) / len(test_words)) if test_words else 0.0

            rows.append({
                "Language": ds_lbl,
                "OOV Words Count": float(len(oov_words)),
                "Total Unique Words in Test": float(len(test_words)),
                "OOV Rate": round(oov_rate, round_to),
            })

        return pd.DataFrame(
            rows,
            columns=["Language", "OOV Words Count", "Total Unique Words in Test", "OOV Rate"]
        )


class EntityTagOOVRate(BaseDatasetHelper):
    """
    Per-tag OOV, computed train→test for each dataset.
    Output rows: Dataset, Tag, OOV Words Count, Total Unique Words in Test, OOV Rate
    """

    def __init__(self, corpora: Dict, tags: Optional[List[str]] = None):
        super().__init__(corpora)
        self.tags = tags or _DEFAULT_TAGS

    def _unique_words_per_tag(self, split_sentences, tags: List[str]) -> Dict[str, set]:
        buckets = {t: set() for t in tags}
        for s in split_sentences:
            for w, t in zip(s["words"], s["tags"]):
                t = self._normalize_tag(t)  # e.g., PERS→PER
                if t in buckets:
                    buckets[t].add(w)
        return buckets

    def generate_df(self, selected_variant: str, round_to: int = 3) -> pd.DataFrame:
        rows: List[dict] = []
        seen_labels = set()

        for ds_key in self.resolve_language_keys(selected_variant):
            ds_lbl = self.lang_label(ds_key)
            if ds_lbl in seen_labels:
                continue
            seen_labels.add(ds_lbl)

            splits = self.corpora[ds_key]["splits"]
            train = splits.get("train")
            test  = splits.get("test")
            if not train or not test:
                # skip silently; or emit zeros if you prefer
                continue

            train_buckets = self._unique_words_per_tag(train, self.tags)
            test_buckets  = self._unique_words_per_tag(test,  self.tags)

            for tag in self.tags:
                test_set  = test_buckets.get(tag, set())
                train_set = train_buckets.get(tag, set())
                oov_set   = test_set - train_set
                oov_rate  = (len(oov_set) / len(test_set)) if test_set else 0.0
                rows.append({
                    "Language": ds_lbl,
                    "Tag": tag,
                    "OOV Words Count": float(len(oov_set)),
                    "Total Unique Words in Test": float(len(test_set)),
                    "OOV Rate": round(oov_rate, round_to),
                })

        return pd.DataFrame(rows, columns=[
            "Language", "Tag", "OOV Words Count", "Total Unique Words in Test", "OOV Rate"
        ])



class TokenisedDatasetStatsHelper(BaseDashDataProcessor):
    """
    Tokens-only stats from DF-based corpora built by BaseDashDataProcessor.build_corpora().
    Expects columns: 'Core Tokens', 'True Labels'.
    Output: rows = metrics, cols = (Language, Split[Train/Test/(Total)]).
    """

    @staticmethod
    def _compute_from_df(df: pd.DataFrame) -> dict:
        if "Core Tokens" not in df.columns or "True Labels" not in df.columns:
            raise KeyError("Expected 'Core Tokens' and 'True Labels' in split DataFrame.")

        total_tokens = int(len(df))
        unique_tokens = int(df["Core Tokens"].nunique())

        is_ne = df["True Labels"].astype(str) != "O"
        ne_tokens = int(is_ne.sum())
        unique_ne_tokens = int(df.loc[is_ne, "Core Tokens"].nunique())

        twr  = (unique_tokens / total_tokens) if total_tokens else 0.0
        ne_prop = (ne_tokens / total_tokens) if total_tokens else 0.0
        tewr = (unique_ne_tokens / ne_tokens) if ne_tokens else 0.0

        return {
            "Total Tokens": total_tokens,
            "Unique Tokens": unique_tokens,
            "NE Tokens": ne_tokens,
            "Unique NE Tokens": unique_ne_tokens,
            "Tokens NE Proportion": ne_prop,
            "TTR": twr,
            "TETR": tewr,
        }

    def generate_df(self, selected_variant: str, include_total: bool = True) -> pd.DataFrame:
        # 1) build corpora for this variant
        self.build_corpora(selected_variant)

        metrics = [
            "Total Tokens", "NE Tokens", "Tokens NE Proportion",
            "TTR", "Unique Tokens", "Unique NE Tokens", "TETR",
        ]
        cols, data = [], {m: [] for m in metrics}

        # 2) iterate the corpora dict (Language → splits)
        for dataset_name, content in self.corpora.items():
            split_dict = content["splits"]
            parts = []

            for split_name, split_df in split_dict.items():  # 'Train' / 'Test'
                stats = self._compute_from_df(split_df)
                cols.append((dataset_name, split_name))
                for m in metrics:
                    data[m].append(stats[m])
                parts.append(split_df[["Core Tokens", "True Labels"]])

            if include_total and parts:
                merged = pd.concat(parts, ignore_index=True)
                totals = self._compute_from_df(merged)
                cols.append((dataset_name, "Total"))
                for m in metrics:
                    data[m].append(totals[m])

        # 3) table shape like your other helpers
        df = pd.DataFrame(data, index=pd.RangeIndex(len(cols))).T
        df.columns = pd.MultiIndex.from_tuples(cols, names=["Language", "Split"])
        return df


class EntityTagTokenTypeDistribution(BaseDashDataProcessor):
    def generate_df(self, selected_variant: str) -> pd.DataFrame:
        self.build_corpora(selected_variant)
        rows = []

        for ds_lbl, content in self.corpora.items():
            for split_key, _, df in self.iter_splits(ds_lbl):
                df_ne = df[df["True Labels"] != "O"].copy()
                if df_ne.empty:
                    continue
                if "Token Ids" not in df_ne.columns:
                    raise KeyError("Expected 'Token Ids' column for token-type analysis.")

                # no tag filter here

                total_ne_tokens = int(len(df_ne))
                all_ne_unique_token_types = int(df_ne["Token Ids"].nunique())

                per_tag = (
                    df_ne.groupby("True Labels", dropna=False)
                        .agg(**{
                            "Total Tokens": ("Token Ids", "count"),
                            "Tag Types": ("Token Ids", pd.Series.nunique),
                        })
                        .reset_index()
                        .rename(columns={"True Labels": "Tag"})
                )

                per_tag["TTR"] = np.where(per_tag["Total Tokens"] > 0,
                                          per_tag["Tag Types"] / per_tag["Total Tokens"], 0.0)
                per_tag["Tokens Proportion"] = np.where(total_ne_tokens > 0,
                                                        per_tag["Total Tokens"] / total_ne_tokens, 0.0)
                per_tag["Type Proportion"] = np.where(all_ne_unique_token_types > 0,
                                                      per_tag["Tag Types"] / all_ne_unique_token_types, 0.0)
                per_tag["Split"] = split_key
                per_tag["Language"] = ds_lbl
                rows.append(per_tag)

        return (pd.concat(rows, ignore_index=True) if rows else
                pd.DataFrame(columns=["Split","Tag","Total Tokens","Tag Types","TTR","Tokens Proportion","Type Proportion","Language"]))

class TokenTypeFrequencyDistribution(BaseDashDataProcessor):
    def generate_df(self, selected_variant: str, round_to: int = 2) -> pd.DataFrame:
        self.build_corpora(selected_variant)
        rows = []

        for ds_lbl, content in self.corpora.items():
            for split_key, _, df in self.iter_splits(ds_lbl):
                df_ne = df[df["True Labels"] != "O"].copy()
                if df_ne.empty:
                    continue
                if "Token Ids" not in df_ne.columns:
                    raise KeyError("Expected 'Token Ids' column for token-type analysis.")

                # no tag filter here

                per_tag_counters = {
                    tag: Counter(g["Token Ids"])
                    for tag, g in df_ne.groupby("True Labels")
                }

                for tag, counter in per_tag_counters.items():
                    freqs = np.fromiter(counter.values(), dtype=np.int64) if counter else np.array([], dtype=np.int64)
                    if freqs.size:
                        mean_f = float(np.mean(freqs))
                        med_f  = float(np.median(freqs))
                        std_f  = float(np.std(freqs, ddof=0))
                    else:
                        mean_f = med_f = std_f = 0.0

                    rows.append({
                        "Language": ds_lbl,
                        "Split": split_key,
                        "Tag": tag,
                        "Mean Frequency": round(mean_f, round_to),
                        "Median Frequency": round(med_f, round_to),
                        "Standard Deviation": round(std_f, round_to),
                    })
        return pd.DataFrame(rows)




class DatasetTokenOOVRate(BaseDashDataProcessor):
    """
    One row per language:
      - OOV Core Tokens Count
      - Total Unique Core Tokens in Test
      - OOV Rate = count / total (decimal, not %)
    """

    def generate_df(self, selected_variant: str, round_to: int = 4) -> pd.DataFrame:
        self.build_corpora(selected_variant)

        rows = []
        for ds_lbl, content in self.corpora.items():
            train = content["splits"]["Train"]
            test  = content["splits"]["Test"]

            # unique sets (all tokens, not per tag)
            train_set = set(train["Core Tokens"].dropna().astype(str).unique())
            test_set  = set(test["Core Tokens"].dropna().astype(str).unique())

            oov = test_set - train_set
            denom = len(test_set)
            rate = (len(oov) / denom) if denom else 0.0

            rows.append({
                "Language": ds_lbl,
                "OOV Core Tokens Count": len(oov),
                "Total Unique Core Tokens in Test": denom,
                "OOV Rate": round(rate, round_to),
            })

        # wide → to match your simple table shape (no splits)
        if not rows:
            return pd.DataFrame(columns=["OOV Core Tokens Count","Total Unique Core Tokens in Test","OOV Rate"])

        df_long = pd.DataFrame(rows)
        df_long.set_index("Dataset", inplace=True)
        # return tidy wide (datasets as columns)
        return df_long.T  # rows = metrics; cols = datasets


class EntityTagTokenOOVRate(BaseDashDataProcessor):
    """
    One row per language × tag:
      - OOV Core Tokens Count
      - Total Unique Core Tokens in Test
      - OOV Rate (decimal)
    """

    @staticmethod
    def _per_tag_sets(df: pd.DataFrame) -> Dict[str, set]:
        """Return {tag -> set(unique Core Tokens)} for NE tags (True Labels != 'O')."""
        if "True Labels" not in df.columns or "Core Tokens" not in df.columns:
            raise KeyError("Expected columns 'True Labels' and 'Core Tokens'.")
        df_ne = df[df["True Labels"] != "O"][["True Labels", "Core Tokens"]].dropna()
        buckets: Dict[str, set] = {}
        for tag, g in df_ne.groupby("True Labels"):
            buckets[tag] = set(g["Core Tokens"].astype(str).unique())
        return buckets

    def generate_df(self, selected_variant: str, round_to: int = 3) -> pd.DataFrame:
        self.build_corpora(selected_variant)
        rows = []

        for ds_lbl, content in self.corpora.items():
            train = content["splits"]["Train"]
            test  = content["splits"]["Test"]

            train_sets = self._per_tag_sets(train)
            test_sets  = self._per_tag_sets(test)

            # Use tags observed in TEST (typical OOV target); include train-only tags if you want union
            tags = sorted(test_sets.keys())

            for tag in tags:
                test_set  = test_sets.get(tag, set())
                train_set = train_sets.get(tag, set())
                oov_set   = test_set - train_set
                denom     = len(test_set)
                rate      = (len(oov_set) / denom) if denom else 0.0

                rows.append({
                    "Language": ds_lbl,
                    "Tag": tag,
                    "OOV Words Count": float(len(oov_set)),                 # keep names for compatibility
                    "Total Unique Words in Test": float(denom),
                    "OOV Rate": round(rate, round_to),                      # decimal (not %)
                })

        return pd.DataFrame(
            rows,
            columns=["Language", "Tag", "OOV Words Count", "Total Unique Words in Test", "OOV Rate"]
        )
    

class WordTypeOverlap(BaseDatasetHelper):
    """
    Compute overlap of unique word *types* across entity tags for sentence-based corpora.

    Options mirror the DF-based class:
      - include_O, tag_set, fill_diagonal, tag_order
    Tag normalisation uses BaseDatasetHelper._normalize_tag (PERS→PER etc.).
    """

    def _tag_sets_for_split(
        self,
        split_sentences: list[dict],
        *,
        include_O: bool,
        allowed_tags: Optional[set]
    ) -> dict[str, set]:
        buckets: dict[str, set] = {}
        for s in split_sentences:
            for w, t in zip(s["words"], s["tags"]):
                t = self._normalize_tag(str(t))
                if not include_O and t == "O":
                    continue
                if allowed_tags is not None and t not in allowed_tags:
                    continue
                buckets.setdefault(t, set()).add(str(w))
        return buckets

    @staticmethod
    def _overlap_matrix(buckets: dict[str, set], *, order: list[str], fill_diagonal: bool) -> pd.DataFrame:
        mat = pd.DataFrame(0, index=order, columns=order, dtype=int)
        for t1 in order:
            for t2 in order:
                if t1 == t2 and fill_diagonal:
                    mat.loc[t1, t2] = 0
                else:
                    mat.loc[t1, t2] = len(buckets.get(t1, set()) & buckets.get(t2, set()))
        return mat

    def _discover_tags_for_dataset(self, ds_key: str, include_O: bool) -> list[str]:
        tags = set()
        for _, _, split_sentences in self.iter_splits(ds_key):
            for s in split_sentences:
                for t in s["tags"]:
                    t = self._normalize_tag(str(t))
                    if not include_O and t == "O":
                        continue
                    tags.add(t)
        return sorted(tags)

    def generate_matrices(
        self,
        selected_variant: str,
        *,
        include_O: bool,
        tag_set: Optional[list[str]] = None,
        fill_diagonal: bool = True,
        tag_order: Optional[list[str]] = None,
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Returns:
          {DatasetLabel -> {SplitLabel -> pd.DataFrame(matrix)}}
        Note: This class expects `self.corpora` in the sentence-based shape you
              already use with BaseDatasetHelper (i.e., not built via BaseDashDataProcessor).
        """
        out: dict[str, dict[str, pd.DataFrame]] = {}

        for ds_key in self.resolve_language_keys(selected_variant):
            ds_lbl = self.lang_label(ds_key)

            # choose tag inventory
            if tag_set is None:
                tags = self._discover_tags_for_dataset(ds_key, include_O=include_O)
            else:
                tags = list(tag_set)
                if not include_O and "O" in tags:
                    tags = [t for t in tags if t != "O"]

            order = tag_order or tags
            out[ds_lbl] = {}

            for _, split_lbl, split_sentences in self.iter_splits(ds_key):
                buckets = self._tag_sets_for_split(
                    split_sentences,
                    include_O=include_O,
                    allowed_tags=set(order),
                )
                mat = self._overlap_matrix(buckets, order=order, fill_diagonal=fill_diagonal)
                out[ds_lbl][split_lbl] = mat

        return out

    def generate_df(
        self,
        selected_variant: str,
        *,
        include_O: bool = True,
        tag_set: Optional[list[str]] = None,
        fill_diagonal: bool = True,
        tag_order: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Returns tidy long form:
          columns = [Language, Split, Tag1, Tag2, Overlap Count]
        """
        mats = self.generate_matrices(
            selected_variant,
            include_O=include_O,
            tag_set=tag_set,
            fill_diagonal=fill_diagonal,
            tag_order=tag_order,
        )
        rows = []
        for ds_lbl, splits in mats.items():
            for split_lbl, mat in splits.items():
                for t1 in mat.index:
                    for t2 in mat.columns:
                        rows.append({
                            "Language": ds_lbl,
                            "Split": split_lbl,
                            "Tag1": t1,
                            "Tag2": t2,
                            "Overlap Count": int(mat.loc[t1, t2]),
                        })
        return pd.DataFrame(rows, columns=["Language","Split","Tag1","Tag2","Overlap Count"])


class TokenTypeOverlap(BaseDashDataProcessor):
    """
    Compute overlap of unique token *types* across entity tags for DF-based corpora
    (produced by BaseDashDataProcessor.build_corpora()).

    Options:
      - include_O: include the 'O' tag as one bucket
      - tag_set:   fixed tag inventory; if None, derive from data per dataset (union of splits)
      - fill_diagonal: set diagonal to 0 (typical for overlap visuals)
      - tag_order: explicit ordering; else sorted discovered tags
    """

    def _tag_sets_for_split(
        self, df: pd.DataFrame, *, include_O: bool, allowed_tags: Optional[set]
    ) -> dict[str, set]:
        if "Core Tokens" not in df.columns or "True Labels" not in df.columns:
            raise KeyError("Expected columns 'Core Tokens' and 'True Labels'.")

        df = df[["Core Tokens", "True Labels"]].dropna().copy()
        df["True Labels"] = df["True Labels"].astype(str)

        if not include_O:
            df = df[df["True Labels"] != "O"]

        if allowed_tags is not None:
            df = df[df["True Labels"].isin(allowed_tags)]

        buckets: dict[str, set] = {}
        for tag, g in df.groupby("True Labels", dropna=False):
            buckets[tag] = set(g["Core Tokens"].astype(str).unique())
        return buckets

    @staticmethod
    def _overlap_matrix(buckets: dict[str, set], *, order: list[str], fill_diagonal: bool) -> pd.DataFrame:
        mat = pd.DataFrame(0, index=order, columns=order, dtype=int)
        for t1 in order:
            for t2 in order:
                if t1 == t2 and fill_diagonal:
                    mat.loc[t1, t2] = 0
                else:
                    s1 = buckets.get(t1, set())
                    s2 = buckets.get(t2, set())
                    mat.loc[t1, t2] = len(s1 & s2)
        return mat

    def _discover_tags_for_dataset(self, ds_lbl: str, include_O: bool) -> list[str]:
        tags = set()
        for split_key, _, df in self.iter_splits(ds_lbl):
            vals = set(df["True Labels"].dropna().astype(str).unique())
            if not include_O:
                vals.discard("O")
            tags |= vals
        return sorted(tags)

    def generate_matrices(
        self,
        selected_variant: str,
        *,
        include_O: bool,
        tag_set: Optional[list[str]] = None,
        fill_diagonal: bool = True,
        tag_order: Optional[list[str]] = None,
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Returns:
          {Language -> {Split -> pd.DataFrame(matrix)}}
        """
        self.build_corpora(selected_variant)

        out: dict[str, dict[str, pd.DataFrame]] = {}
        for ds_lbl, content in self.corpora.items():
            # choose tag inventory
            if tag_set is None:
                tags = self._discover_tags_for_dataset(ds_lbl, include_O=include_O)
            else:
                tags = list(tag_set)
                if not include_O and "O" in tags:
                    tags = [t for t in tags if t != "O"]

            order = tag_order or tags
            out[ds_lbl] = {}

            for split_key, _, df in self.iter_splits(ds_lbl):
                buckets = self._tag_sets_for_split(df, include_O=include_O, allowed_tags=set(order))
                mat = self._overlap_matrix(buckets, order=order, fill_diagonal=fill_diagonal)
                out[ds_lbl][split_key] = mat

        return out

    def generate_df(
        self,
        selected_variant: str,
        *,
        include_O: bool = True,
        tag_set: Optional[list[str]] = None,
        fill_diagonal: bool = True,
        tag_order: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Returns tidy long form:
          columns = [Language, Split, Tag1, Tag2, Overlap Count]
        """
        mats = self.generate_matrices(
            selected_variant,
            include_O=include_O,
            tag_set=tag_set,
            fill_diagonal=fill_diagonal,
            tag_order=tag_order,
        )
        rows = []
        for ds_lbl, splits in mats.items():
            for split_lbl, mat in splits.items():
                for t1 in mat.index:
                    for t2 in mat.columns:
                        rows.append({
                            "Language": ds_lbl,
                            "Split": split_lbl,
                            "Tag1": t1,
                            "Tag2": t2,
                            "Overlap Count": int(mat.loc[t1, t2]),
                        })
        return pd.DataFrame(rows, columns=["Language","Split","Tag1","Tag2","Overlap Count"])




class TokenizationRateHelper(BaseDashDataProcessor):
    """
    Generates tokenisation rate per entity tag.
    Outputs Language × Split × Tag with Mean, Std, and Text Label.
    """

    COL = "Tokenization Rate"

    @staticmethod
    def _summarise(split_df: pd.DataFrame, value_col: str, round_to: int) -> pd.DataFrame:
        if value_col not in split_df.columns or "True Labels" not in split_df.columns:
            return pd.DataFrame(columns=["Tag", "Mean Value", "Std Dev", "Text Label"])

        df = split_df[["True Labels", value_col]].copy()
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col])

        if df.empty:
            return pd.DataFrame(columns=["Tag", "Mean Value", "Std Dev", "Text Label"])

        g = (
            df.groupby("True Labels", as_index=False)[value_col]
              .agg(**{"Mean Value": "mean", "Std Dev": "std"})
              .rename(columns={"True Labels": "Tag"})
        )
        g["Std Dev"] = g["Std Dev"].fillna(0.0)
        g["Mean Value"] = g["Mean Value"].round(round_to)
        g["Std Dev"]    = g["Std Dev"].round(round_to)

        g["Text Label"] = g.apply(
            lambda r: f"{r['Mean Value']:.{round_to}f} <br>±<br> {r['Std Dev']:.{round_to}f}",
            axis=1
        )
        return g[["Tag", "Mean Value", "Std Dev", "Text Label"]]

    def generate_df(self, selected_variant: str, round_to: int = 3) -> pd.DataFrame:
        self.build_corpora(selected_variant)

        rows = []
        for ds_lbl, content in self.corpora.items():
            for split_key, _, split_df in self.iter_splits(ds_lbl, only_test=True):
                out = self._summarise(split_df, self.COL, round_to)
                if not out.empty:
                    out["Language"] = ds_lbl
                    out["Split"]   = split_key
                    rows.append(out)

        if not rows:
            return pd.DataFrame(columns=["Language", "Split", "Tag", "Mean Value", "Std Dev", "Text Label"])

        return pd.concat(rows, ignore_index=True)[
            ["Language", "Split", "Tag", "Mean Value", "Std Dev", "Text Label"]
        ]

    


class AmbiguityHelper(BaseDashDataProcessor):
    """
    Merge Token and Word ambiguity at entity level, per Language × Split.
    Output columns:
      Language, Split, Tag, Level ('Token Level'|'Word Level'),
      Mean Value, Std Dev, Text Label
    """
    TOKEN_COL = "Token Ambiguity"
    WORD_COL  = "Word Ambiguity"

    @staticmethod
    def _summarise(split_df: pd.DataFrame, value_col: str, level_name: str, round_to: int) -> pd.DataFrame:
        if value_col not in split_df.columns:
            return pd.DataFrame(columns=["Tag", "Level", "Mean Value", "Std Dev", "Text Label"])

        df = split_df.loc[split_df[value_col] != -1, ["True Labels", value_col]].copy()
        if df.empty:
            return pd.DataFrame(columns=["Tag", "Level", "Mean Value", "Std Dev", "Text Label"])

        g = (
            df.groupby("True Labels", as_index=False)[value_col]
              .agg(["mean", "std"])
              .reset_index()
              .rename(columns={"True Labels": "Tag", "mean": "Mean Value", "std": "Std Dev"})
        )
        g["Mean Value"] = pd.to_numeric(g["Mean Value"], errors="coerce").round(round_to)
        g["Std Dev"]    = pd.to_numeric(g["Std Dev"],    errors="coerce").round(round_to)
        g["Level"]      = level_name
        g["Text Label"] = g.apply(
            lambda r: f"{(r['Mean Value'] if pd.notna(r['Mean Value']) else 0):.{round_to}f}"
                      f" <br>±<br>"
                      f"{(r['Std Dev'] if pd.notna(r['Std Dev']) else 0):.{round_to}f}",
            axis=1
        )
        return g[["Tag", "Level", "Mean Value", "Std Dev", "Text Label"]]

    def generate_df(self, selected_variant: str, round_to: int = 3) -> pd.DataFrame:
        self.build_corpora(selected_variant)

        rows = []
        for ds_lbl, content in self.corpora.items():
            for split_key, _, split_df in self.iter_splits(ds_lbl, only_test=True):
                # Token-level ambiguity
                tok = self._summarise(split_df, self.TOKEN_COL, "Token Level", round_to)
                if not tok.empty:
                    tok["Language"] = ds_lbl
                    tok["Split"]   = split_key
                    rows.append(tok)

                # Word-level ambiguity
                wrd = self._summarise(split_df, self.WORD_COL, "Word Level", round_to)
                if not wrd.empty:
                    wrd["Language"] = ds_lbl
                    wrd["Split"]   = split_key
                    rows.append(wrd)

        if not rows:
            return pd.DataFrame(columns=[
                "Language", "Split", "Tag", "Level", "Mean Value", "Std Dev", "Text Label"
            ])

        out = pd.concat(rows, ignore_index=True)
        return out[["Language", "Split", "Tag", "Level", "Mean Value", "Std Dev", "Text Label"]]

class ConsistencyHelper(BaseDashDataProcessor):
    """
    Merge Consistency and Inconsistency ratios at entity level, per Language × Split.
    Output columns:
      Language, Split, Tag, Level ('Consistency Ratio'|'Inconsistency Ratio'),
      Mean Value, Std Dev, Text Label
    """
    CONSISTENCY_COL    = "Consistency Ratio"
    INCONSISTENCY_COL  = "Inconsistency Ratio"

    @staticmethod
    def _summarise(split_df: pd.DataFrame, value_col: str, level_name: str, round_to: int) -> pd.DataFrame:
        # guard: column present?
        if value_col not in split_df.columns or "True Labels" not in split_df.columns:
            return pd.DataFrame(columns=["Tag", "Level", "Mean Value", "Std Dev", "Text Label"])

        # keep valid numeric values only
        df = split_df[["True Labels", value_col]].copy()
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col])

        if df.empty:
            return pd.DataFrame(columns=["Tag", "Level", "Mean Value", "Std Dev", "Text Label"])

        # aggregate by tag
        g = (
            df.groupby("True Labels", as_index=False)[value_col]
              .agg(**{"Mean Value": "mean", "Std Dev": "std"})  # pandas std -> ddof=1
              .rename(columns={"True Labels": "Tag"})
        )

        # handle single-sample std (NaN) cleanly
        g["Std Dev"] = g["Std Dev"].fillna(0.0)

        # rounding
        g["Mean Value"] = pd.to_numeric(g["Mean Value"], errors="coerce").round(round_to)
        g["Std Dev"]    = pd.to_numeric(g["Std Dev"],    errors="coerce").round(round_to)

        # annotate display fields
        g["Level"]      = level_name
        g["Text Label"] = g.apply(
            lambda r: f"{(r['Mean Value'] if pd.notna(r['Mean Value']) else 0):.{round_to}f}"
                      f" <br>±<br>"
                      f"{(r['Std Dev'] if pd.notna(r['Std Dev']) else 0):.{round_to}f}",
            axis=1
        )
        return g[["Tag", "Level", "Mean Value", "Std Dev", "Text Label"]]

    def generate_df(self, selected_variant: str, round_to: int = 3) -> pd.DataFrame:
        self.build_corpora(selected_variant)

        rows = []
        for ds_lbl, content in self.corpora.items():
            for split_key, _, split_df in self.iter_splits(ds_lbl, only_test=True):
                # Consistency
                con = self._summarise(split_df, self.CONSISTENCY_COL, "Consistency Ratio", round_to)
                if not con.empty:
                    con["Language"] = ds_lbl
                    con["Split"]   = split_key
                    rows.append(con)

                # Inconsistency
                inc = self._summarise(split_df, self.INCONSISTENCY_COL, "Inconsistency Ratio", round_to)
                if not inc.empty:
                    inc["Language"] = ds_lbl
                    inc["Split"]   = split_key
                    rows.append(inc)

        if not rows:
            return pd.DataFrame(columns=[
                "Language", "Split", "Tag", "Level", "Mean Value", "Std Dev", "Text Label"
            ])

        out = pd.concat(rows, ignore_index=True)
        return out[["Language", "Split", "Tag", "Level", "Mean Value", "Std Dev", "Text Label"]]


class LossHelper(BaseDashDataProcessor):
    COL = "Loss Values"

    @staticmethod
    def _summarise(split_df: pd.DataFrame, value_col: str, round_to: int) -> pd.DataFrame:
        if value_col not in split_df.columns or "True Labels" not in split_df.columns:
            return pd.DataFrame(columns=["Tag", "Mean Value", "Std Dev", "Text Label"])

        df = split_df[["True Labels", value_col]].copy()
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col])
        if df.empty:
            return pd.DataFrame(columns=["Tag", "Mean Value", "Std Dev", "Text Label"])

        g = (df.groupby("True Labels", as_index=False)[value_col]
               .agg(**{"Mean Value": "mean", "Std Dev": "std"})
               .rename(columns={"True Labels": "Tag"}))
        g["Std Dev"] = g["Std Dev"].fillna(0.0)
        g["Mean Value"] = g["Mean Value"].round(round_to)
        g["Std Dev"]    = g["Std Dev"].round(round_to)
        g["Text Label"] = g.apply(
            lambda r: f"{r['Mean Value']:.{round_to}f} <br>±<br> {r['Std Dev']:.{round_to}f}", axis=1
        )
        return g[["Tag", "Mean Value", "Std Dev", "Text Label"]]

    def generate_df(self, selected_variant: str, round_to: int = 3) -> pd.DataFrame:
        self.build_corpora(selected_variant)
        rows = []
        for ds_lbl, content in self.corpora.items():
            for split_key, _, split_df in self.iter_splits(ds_lbl, only_test=True):
                out = self._summarise(split_df, self.COL, round_to)
                if not out.empty:
                    out["Language"] = ds_lbl
                    out["Split"]   = split_key
                    rows.append(out)
        return (pd.concat(rows, ignore_index=True)
                if rows else
                pd.DataFrame(columns=["Language","Split","Tag","Mean Value","Std Dev","Text Label"]))



class SilhouetteHelper(BaseDashDataProcessor):
    """
    Per-tag silhouette scores at the representation layer.
    Expects columns on split DF: 'True Silhouette', 'Pred Silhouette', 'True Labels'
    Emits tidy rows: Language, Split, Tag, Level, Mean Value, Std Dev, Text Label
    """
    TRUE_COL = "True Silhouette"
    PRED_COL = "Pred Silhouette"

    @staticmethod
    def _summarise(split_df: pd.DataFrame, value_col: str, level_name: str, round_to: int) -> pd.DataFrame:
        if value_col not in split_df.columns or "True Labels" not in split_df.columns:
            return pd.DataFrame(columns=["Tag", "Level", "Mean Value", "Std Dev", "Text Label"])

        df = split_df.loc[:, ["True Labels", value_col]].copy()
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col])
        if df.empty:
            return pd.DataFrame(columns=["Tag", "Level", "Mean Value", "Std Dev", "Text Label"])

        g = (
            df.groupby("True Labels", as_index=False)[value_col]
              .agg(**{"Mean Value": "mean", "Std Dev": "std"})
              .rename(columns={"True Labels": "Tag"})
        )
        g["Std Dev"]    = g["Std Dev"].fillna(0.0)
        g["Mean Value"] = pd.to_numeric(g["Mean Value"], errors="coerce").round(round_to)
        g["Std Dev"]    = pd.to_numeric(g["Std Dev"],    errors="coerce").round(round_to)
        g["Level"]      = level_name
        g["Text Label"] = g.apply(
            lambda r: f"{(r['Mean Value'] if pd.notna(r['Mean Value']) else 0):.{round_to}f}"
                      f" <br>±<br>"
                      f"{(r['Std Dev'] if pd.notna(r['Std Dev']) else 0):.{round_to}f}",
            axis=1
        )
        return g[["Tag", "Level", "Mean Value", "Std Dev", "Text Label"]]

    def generate_df(self, selected_variant: str, round_to: int = 3) -> pd.DataFrame:
        # Use only the Test split for fair comparison (like Loss/Ambiguity/Consistency)
        self.build_corpora(selected_variant)

        rows = []
        for lang_lbl, content in self.corpora.items():
            for split_key, _, split_df in self.iter_splits(lang_lbl, only_test=True):
                # True silhouettes
                true_df = self._summarise(split_df, self.TRUE_COL, "True Silhouette", round_to)
                if not true_df.empty:
                    true_df["Language"] = lang_lbl
                    true_df["Split"]    = split_key
                    rows.append(true_df)

                # Pred silhouettes
                pred_df = self._summarise(split_df, self.PRED_COL, "Pred Silhouette", round_to)
                if not pred_df.empty:
                    pred_df["Language"] = lang_lbl
                    pred_df["Split"]    = split_key
                    rows.append(pred_df)

        if not rows:
            return pd.DataFrame(columns=["Language","Split","Tag","Level","Mean Value","Std Dev","Text Label"])

        out = pd.concat(rows, ignore_index=True)
        return out[["Language","Split","Tag","Level","Mean Value","Std Dev","Text Label"]]


class PredictionUncertaintyHelper(BaseDashDataProcessor):
    """
    Entity-level prediction uncertainty per Language × Split × Tag,
    optionally split into Correct vs Error based on 'Agreements' boolean.
    Produces: Language, Split, Tag, Type, Mean Value, Std Dev, Text Label
    """
    COL = "Prediction Uncertainty"

    @staticmethod
    def _summarise(split_df: pd.DataFrame, value_col: str, round_to: int) -> pd.DataFrame:
        need = {"True Labels", value_col}
        if not need.issubset(split_df.columns):
            return pd.DataFrame(columns=["Tag", "Mean Value", "Std Dev", "Text Label"])

        df = split_df.loc[:, ["True Labels", value_col]].copy()
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col])
        if df.empty:
            return pd.DataFrame(columns=["Tag", "Mean Value", "Std Dev", "Text Label"])

        g = (
            df.groupby("True Labels", as_index=False)[value_col]
              .agg(**{"Mean Value": "mean", "Std Dev": "std"})
              .rename(columns={"True Labels": "Tag"})
        )
        g["Std Dev"] = g["Std Dev"].fillna(0.0)
        g["Mean Value"] = pd.to_numeric(g["Mean Value"], errors="coerce").round(round_to)
        g["Std Dev"]    = pd.to_numeric(g["Std Dev"],    errors="coerce").round(round_to)
        g["Text Label"] = g.apply(
            lambda r: f"{(r['Mean Value'] if pd.notna(r['Mean Value']) else 0):.{round_to}f}"
                      f" <br>±<br>"
                      f"{(r['Std Dev'] if pd.notna(r['Std Dev']) else 0):.{round_to}f}",
            axis=1
        )
        return g[["Tag", "Mean Value", "Std Dev", "Text Label"]]

    def _slice_by_type(self, df: pd.DataFrame, want: str) -> pd.DataFrame:
        """
        want in {'all','correct','error'}; requires 'Agreements' if not 'all'.
        """
        if want == "all":
            return df
        if "Agreements" not in df.columns:
            # If Agreements missing, return empty so we don't mislead
            return df.iloc[0:0].copy()
        mask = (df["Agreements"] == True) if want == "correct" else (df["Agreements"] == False)
        return df.loc[mask].copy()

    def generate_df(
        self,
        selected_variant: str,
        round_to: int = 3,
        mode: str = "both",     # 'all' | 'correct' | 'error' | 'both'
    ) -> pd.DataFrame:
        """
        mode='both' returns Correct and Error stacked with a 'Type' column;
        others return a single Type.
        """
        self.build_corpora(selected_variant)

        wanted_types = (
            ["correct", "error"] if mode == "both" else
            ([mode] if mode in {"all", "correct", "error"} else ["all"])
        )

        rows = []
        for lang_lbl, content in self.corpora.items():
            for split_key, _, split_df in self.iter_splits(lang_lbl, only_test=True):
                for t in wanted_types:
                    sliced = self._slice_by_type(split_df, t)
                    if sliced.empty:
                        continue
                    out = self._summarise(sliced, self.COL, round_to)
                    if out.empty:
                        continue
                    out["Language"] = lang_lbl
                    out["Split"]    = split_key
                    out["Type"]     = (
                        "All" if t == "all" else ("Correct" if t == "correct" else "Error")
                    )
                    rows.append(out)

        if not rows:
            return pd.DataFrame(columns=["Language","Split","Tag","Type","Mean Value","Std Dev","Text Label"])

        df = pd.concat(rows, ignore_index=True)
        return df[["Language","Split","Tag","Type","Mean Value","Std Dev","Text Label"]]

class PerClassConfidenceHelper(BaseDashDataProcessor):
    TAG_CONF_COLS = [
        "O Confidence",
        "B-LOC Confidence", "I-LOC Confidence",
        "B-PER Confidence", "I-PER Confidence",
        "B-ORG Confidence", "I-ORG Confidence",
        "B-MISC Confidence", "I-MISC Confidence",
    ]

    def generate_df(self, selected_variant: str, *, round_to: int = 3, include_all: bool = False) -> pd.DataFrame:
        """
        Returns long tidy:
          Language, Split, Tag, Mean Value, Std Dev, Text Label, Type

        Always computes 'Correct' and 'Error'. If include_all=True, also adds 'All'.
        """
        self.build_corpora(selected_variant)
        rows = []

        def _agg_block(df_block: pd.DataFrame, lang_lbl: str, split_key: str, type_label: str):
            avail = [c for c in self.TAG_CONF_COLS if c in df_block.columns]
            if not avail:
                return None

            long = df_block.loc[:, avail].melt(var_name="Class", value_name="value").dropna()
            if long.empty:
                return None

            # derive Tag without the " Confidence" suffix
            long["Tag"] = long["Class"].str.replace(" Confidence", "", regex=False)

            g = (
                long.groupby("Tag")["value"]
                    .agg(**{"Mean Value": "mean", "Std Dev": "std"})
                    .reset_index()
            )
            g["Std Dev"]    = g["Std Dev"].fillna(0.0).round(round_to)
            g["Mean Value"] = g["Mean Value"].round(round_to)
            g["Text Label"] = g.apply(
                lambda r: f"{r['Mean Value']:.{round_to}f} <br>±<br> {r['Std Dev']:.{round_to}f}",
                axis=1,
            )
            g["Language"] = lang_lbl
            g["Split"]    = split_key
            g["Type"]     = type_label
            return g[["Language", "Split", "Tag", "Mean Value", "Std Dev", "Text Label", "Type"]]

        for lang_lbl, content in self.corpora.items():
            for split_key, _, split_df in self.iter_splits(lang_lbl, only_test=True):
                # If Agreements column exists, compute Correct + Error
                if "Agreements" in split_df.columns:
                    correct_df = split_df.loc[split_df["Agreements"] == True]
                    error_df   = split_df.loc[split_df["Agreements"] == False]

                    blk = _agg_block(correct_df, lang_lbl, split_key, "Correct")
                    if blk is not None:
                        rows.append(blk)

                    blk = _agg_block(error_df, lang_lbl, split_key, "Error")
                    if blk is not None:
                        rows.append(blk)

                    if include_all:
                        blk = _agg_block(split_df, lang_lbl, split_key, "All")
                        if blk is not None:
                            rows.append(blk)
                else:
                    # No Agreements column → only “All”
                    blk = _agg_block(split_df, lang_lbl, split_key, "All")
                    if blk is not None:
                        rows.append(blk)

        return (
            pd.concat(rows, ignore_index=True)
            if rows else
            pd.DataFrame(columns=["Language", "Split", "Tag", "Mean Value", "Std Dev", "Text Label", "Type"])
        )
class TokenConfidenceHelper(BaseDashDataProcessor):
    """
    Token-level confidence aggregated by raw BIO tag from 'True Labels'.
    Mirrors PerClassConfidenceHelper:
      - Splits by Agreements -> Correct / Error (and optional All)
      - Works per (Language, Split)
      - Returns long tidy with the same column schema:
        Language, Split, Tag, Mean Value, Std Dev, Text Label, Type
    """

    CONF_COL = "Token Confidence"
    TRUE_COL = "True Labels"

    def generate_df(self, selected_variant: str, *, round_to: int = 3, include_all: bool = False) -> pd.DataFrame:
        """
        Returns long tidy:
          Language, Split, Tag, Mean Value, Std Dev, Text Label, Type

        Always computes 'Correct' and 'Error' when 'Agreements' exists.
        If include_all=True, also adds 'All'. If 'Agreements' is missing, returns only 'All'.
        """
        self.build_corpora(selected_variant)
        rows: List[pd.DataFrame] = []

        def _agg_block(df_block: pd.DataFrame, lang_lbl: str, split_key: str, type_label: str) -> Optional[pd.DataFrame]:
            # strict column check
            for col in (self.CONF_COL, self.TRUE_COL):
                if col not in df_block.columns:
                    return None

            # keep only needed cols, drop NA confidences
            work = df_block[[self.TRUE_COL, self.CONF_COL]].dropna(subset=[self.CONF_COL])
            if work.empty:
                return None

            # group by raw BIO tag from True Labels (no collapsing)
            g = (
                work.groupby(self.TRUE_COL)[self.CONF_COL]
                    .agg(**{"Mean Value": "mean", "Std Dev": "std"})
                    .reset_index()
                    .rename(columns={self.TRUE_COL: "Tag"})
            )

            # format like your PerClass helper
            g["Std Dev"]    = g["Std Dev"].fillna(0.0).round(round_to)
            g["Mean Value"] = g["Mean Value"].round(round_to)
            g["Text Label"] = g.apply(
                lambda r: f"{r['Mean Value']:.{round_to}f} <br>±<br> {r['Std Dev']:.{round_to}f}",
                axis=1,
            )
            g["Language"] = lang_lbl
            g["Split"]    = split_key
            g["Type"]     = type_label

            return g[["Language", "Split", "Tag", "Mean Value", "Std Dev", "Text Label", "Type"]]

        for lang_lbl, _content in self.corpora.items():
            # mirror your behavior: test-only
            for split_key, _, split_df in self.iter_splits(lang_lbl, only_test=True):
                if self.CONF_COL not in split_df.columns or self.TRUE_COL not in split_df.columns:
                    # if required columns missing, skip this split (or raise—your call)
                    # raise ValueError(f"Missing required columns in split {lang_lbl}/{split_key}")
                    continue

                if "Agreements" in split_df.columns:
                    correct_df = split_df.loc[split_df["Agreements"] == True]
                    error_df   = split_df.loc[split_df["Agreements"] == False]

                    blk = _agg_block(correct_df, lang_lbl, split_key, "Correct")
                    if blk is not None:
                        rows.append(blk)

                    blk = _agg_block(error_df, lang_lbl, split_key, "Error")
                    if blk is not None:
                        rows.append(blk)

                    if include_all:
                        blk = _agg_block(split_df, lang_lbl, split_key, "All")
                        if blk is not None:
                            rows.append(blk)
                else:
                    blk = _agg_block(split_df, lang_lbl, split_key, "All")
                    if blk is not None:
                        rows.append(blk)

        return (
            pd.concat(rows, ignore_index=True)
            if rows else
            pd.DataFrame(columns=["Language", "Split", "Tag", "Mean Value", "Std Dev", "Text Label", "Type"])
        )


import pandas as pd
from typing import Literal, List

AggType = Literal["mean", "sum"]

class ConfidenceConfusionHelper(BaseDashDataProcessor):
    """
    Prepare confidence-confusion as a long DF compatible with plot_overlap_heatmaps().
    - Filters to misclassified tokens (Agreements == False)
    - Aggregates Token Confidence per (Language, True Labels, Predicted Labels)
    - Exposes aggregation as 'mean' (default) or 'sum'
    Output columns:
      Language, Tag1, Tag2, Overlap Count, Count
    """

    CONF_COL = "Token Confidence"
    TRUE_COL = "True Labels"
    PRED_COL = "Pred Labels"
    LANG_COL = "Language"
    AGR_COL  = "Agreements"

    def _have_required(self, df: pd.DataFrame) -> bool:
        need = {self.CONF_COL, self.TRUE_COL, self.PRED_COL, self.AGR_COL}
        return need.issubset(df.columns)

    def generate_df(
        self,
        selected_variant: str,
        *,
        agg: AggType = "mean",
        round_to: int = 2,
        only_test: bool = True,
    ) -> pd.DataFrame:
        """
        Returns a long dataframe ready for plot_overlap_heatmaps():
          Language, Tag1, Tag2, Overlap Count, Count
        """
        self.build_corpora(selected_variant)
        out_frames: List[pd.DataFrame] = []

        for lang_lbl, _content in self.corpora.items():
            for split_key, _, split_df in self.iter_splits(lang_lbl, only_test=only_test):
                if not self._have_required(split_df):
                    continue

                df = split_df.copy()
                if self.LANG_COL not in df.columns:
                    df[self.LANG_COL] = lang_lbl
                # errors only
                df = df[df[self.AGR_COL] == False]
                df = df[[self.LANG_COL, self.TRUE_COL, self.PRED_COL, self.CONF_COL]].dropna(subset=[self.CONF_COL])
                if df.empty:
                    continue

                if agg == "sum":
                    g = (
                        df.groupby([self.LANG_COL, self.TRUE_COL, self.PRED_COL])[self.CONF_COL]
                          .agg(**{"Overlap Count": "sum", "Count": "count"})
                          .reset_index()
                    )
                    g["Overlap Count"] = g["Overlap Count"].round(round_to)
                else:  # mean
                    g = (
                        df.groupby([self.LANG_COL, self.TRUE_COL, self.PRED_COL])[self.CONF_COL]
                          .agg(**{"Overlap Count": "mean", "Count": "count"})
                          .reset_index()
                    )
                    g["Overlap Count"] = g["Overlap Count"].round(round_to)

                # rename to Tag1/Tag2 for your heatmap util
                g = g.rename(columns={self.TRUE_COL: "Tag1", self.PRED_COL: "Tag2"})
                out_frames.append(g)

        if not out_frames:
            return pd.DataFrame(columns=["Language", "Tag1", "Tag2", "Overlap Count", "Count"])

        return pd.concat(out_frames, ignore_index=True)





class TokenVsEntityOverallHelper(BaseDashDataProcessor):
    """
    Build a wide table:
      rows   = Precision, Recall, F1
      cols   = MultiIndex (Language, Level) where Level in {"Token-Level","Entity-Level"}
      Token  = macro row from token_report
      Entity = micro row from entity_non_strict_report (IOB1 / non-strict)
    """

    @staticmethod
    def _pick_cols(df: pd.DataFrame) -> Tuple[str, str, str]:
        cols_lower = {c.lower(): c for c in df.columns}
        p = cols_lower.get("precision", "Precision")
        r = cols_lower.get("recall", "Recall")
        f = cols_lower.get("f1", cols_lower.get("f1-score", "F1-score"))
        return p, r, f

    @staticmethod
    def _extract_metrics_row(df: pd.DataFrame, tag_value: str) -> Dict[str, float]:
        if df is None or df.empty:
            return {"Precision": 0.0, "Recall": 0.0, "F1-score": 0.0}

        # find tag column robustly
        tag_col = None
        for c in df.columns:
            if c.lower() == "tag":
                tag_col = c
                break
        if tag_col is None:
            return {"Precision": 0.0, "Recall": 0.0, "F1-score": 0.0}

        # row with requested tag (case-insensitive)
        row = df.loc[df[tag_col].astype(str).str.lower() == tag_value.lower()]
        if row.empty:
            return {"Precision": 0.0, "Recall": 0.0, "F1": 0.0}

        pcol, rcol, fcol = TokenVsEntityOverallHelper._pick_cols(df)
        p = pd.to_numeric(row.iloc[0][pcol], errors="coerce")
        r = pd.to_numeric(row.iloc[0][rcol], errors="coerce")
        f = pd.to_numeric(row.iloc[0][fcol], errors="coerce")
        return {
            "Precision": float(p) if pd.notna(p) else 0.0,
            "Recall":    float(r) if pd.notna(r) else 0.0,
            "F1-score":        float(f) if pd.notna(f) else 0.0,
        }

    def generate_df(self, selected_variant: str, round_to: int = 4) -> pd.DataFrame:
        variant_keys = self._resolve_keys_for_variant(selected_variant)

        cols: List[tuple] = []
        data: Dict[str, List[float]] = {k: [] for k in ROW_ORDER}

        for vkey in variant_keys:
            if vkey not in self.dash_data:
                continue

            lang_lbl = self.ds_label(vkey)  # e.g., Arabic / English
            content  = self.dash_data[vkey]

            # Token = macro row
            tok_stats = self._extract_metrics_row(getattr(content, "token_report", None), "macro")

            # Entity (IOB1 / non-strict) = micro row
            ent_stats = self._extract_metrics_row(getattr(content, "entity_non_strict_report", None), "micro")

            # Append token column
            for metric in ROW_ORDER:
                data[metric].append(round(float(tok_stats[metric]), round_to))
            cols.append((lang_lbl, "Token-Level"))

            # Append entity column
            for metric in ROW_ORDER:
                data[metric].append(round(float(ent_stats[metric]), round_to))
            cols.append((lang_lbl, "Entity-Level"))

        out = pd.DataFrame(data, index=pd.RangeIndex(len(cols))).T
        out.columns = pd.MultiIndex.from_tuples(cols, names=["Language", "Level"])
        out = out.reindex(ROW_ORDER)
        return out



class EntitySchemesOverallHelper(BaseDashDataProcessor):
    """
    Wide table:
      rows   = Precision, Recall, F1
      cols   = MultiIndex (Language, Scheme) where
               Scheme ∈ {"IOB1 (Non-Strict)", "IOB2 (Strict)"}
    Values:
      - For each model/language, take the **micro** row from:
          * entity_non_strict_report  → IOB1 (Non-Strict)
          * entity_strict_report      → IOB2 (Strict)
    """

    @staticmethod
    def _pick_cols(df: pd.DataFrame) -> Tuple[str, str, str]:
        cols_lower = {c.lower(): c for c in df.columns}
        p = cols_lower.get("precision", "Precision")
        r = cols_lower.get("recall", "Recall")
        f = cols_lower.get("f1", cols_lower.get("f1-score", "F1"))
        return p, r, f

    @staticmethod
    def _extract_micro(df: pd.DataFrame) -> Dict[str, float]:
        if df is None or df.empty:
            return {"Precision": 0.0, "Recall": 0.0, "F1-score": 0.0}

        # find 'Tag' column
        tag_col = None
        for c in df.columns:
            if c.lower() == "tag":
                tag_col = c
                break
        if tag_col is None:
            return {"Precision": 0.0, "Recall": 0.0, "F1-score": 0.0}

        row = df.loc[df[tag_col].astype(str).str.lower() == "micro"]
        if row.empty:
            return {"Precision": 0.0, "Recall": 0.0, "F1-score": 0.0}

        pcol, rcol, fcol = EntitySchemesOverallHelper._pick_cols(df)
        p = pd.to_numeric(row.iloc[0][pcol], errors="coerce")
        r = pd.to_numeric(row.iloc[0][rcol], errors="coerce")
        f = pd.to_numeric(row.iloc[0][fcol], errors="coerce")
        return {
            "Precision": float(p) if pd.notna(p) else 0.0,
            "Recall":    float(r) if pd.notna(r) else 0.0,
            "F1-score":        float(f) if pd.notna(f) else 0.0,
        }

    def generate_df(self, selected_variant: str, round_to: int = 4) -> pd.DataFrame:
        variant_keys = self._resolve_keys_for_variant(selected_variant)

        cols: List[tuple] = []
        data: Dict[str, List[float]] = {k: [] for k in ROW_ORDER}

        for vkey in variant_keys:
            if vkey not in self.dash_data:
                continue

            lang_lbl = self.ds_label(vkey)  # e.g., Arabic / English
            content  = self.dash_data[vkey]

            iob1 = self._extract_micro(getattr(content, "entity_non_strict_report", None))
            iob2 = self._extract_micro(getattr(content, "entity_strict_report", None))

            # IOB1 (Non-Strict)
            for metric in ROW_ORDER:
                data[metric].append(round(float(iob1[metric]), round_to))
            cols.append((lang_lbl, "IOB1 (Non-Strict)"))

            # IOB2 (Strict)
            for metric in ROW_ORDER:
                data[metric].append(round(float(iob2[metric]), round_to))
            cols.append((lang_lbl, "IOB2 (Strict)"))

        out = pd.DataFrame(data, index=pd.RangeIndex(len(cols))).T
        out.columns = pd.MultiIndex.from_tuples(cols, names=["Language", "Scheme"])
        out = out.reindex(ROW_ORDER)
        return out



class EntitySpanF1Helper(BaseDashDataProcessor):
    """
    Extract F1 scores per entity type from both IOB1 and IOB2 reports.
    Output DataFrame columns: Language, Scheme, Tag, F1
    """

    def generate_df(self, selected_variant: str, round_to: int = 3) -> pd.DataFrame:
        self.build_corpora(selected_variant)
        rows = []

        for ds_lbl, content in self.dash_data.items():
            lang_lbl = self.ds_label(ds_lbl)

            # IOB1 (non-strict)
            # iob1 = content.entity_non_strict_report.copy()
            iob1 = self.normalize_spans(content.entity_non_strict_report.copy())
            iob1 = iob1[~iob1["Tag"].isin(["micro", "macro", "weighted"])]
            for _, row in iob1.iterrows():
                rows.append({
                    "Language": lang_lbl,
                    "Scheme": "IOB1",
                    "Tag": row["Tag"],
                    "F1-score": round(float(row["F1"]), round_to)
                })

            # IOB2 (strict)
            iob2 = self.normalize_spans(content.entity_strict_report.copy())
            iob2 = iob2[~iob2["Tag"].isin(["micro", "macro", "weighted"])]
            for _, row in iob2.iterrows():
                rows.append({
                    "Language": lang_lbl,
                    "Scheme": "IOB2",
                    "Tag": row["Tag"],
                    "F1-score": round(float(row["F1"]), round_to)
                })

        return pd.DataFrame(rows)


class EntitySpanPRHelper(BaseDashDataProcessor):
    """
    Extract Precision & Recall per entity type from both IOB1 and IOB2 reports.
    Output columns: Language, Scheme, Tag, Metric, Value
    """

    def generate_df(self, selected_variant: str, round_to: int = 3) -> pd.DataFrame:
        rows = []

        # use the same variant resolution pattern as elsewhere
        for vkey in self._resolve_keys_for_variant(selected_variant):
            if vkey not in self.dash_data:
                continue
            lang_lbl = self.ds_label(vkey)
            content  = self.dash_data[vkey]

            # IOB1 (non-strict)
            iob1 = self.normalize_spans(content.entity_non_strict_report.copy())
            iob1 = iob1[~iob1["Tag"].isin(["micro", "macro", "weighted"])]
            for _, r in iob1.iterrows():
                rows.append({
                    "Language": lang_lbl,
                    "Scheme":   "IOB1",
                    "Tag":      r["Tag"],
                    "Metric":   "Precision",
                    "Value":    round(float(r["Precision"]), round_to),
                })
                rows.append({
                    "Language": lang_lbl,
                    "Scheme":   "IOB1",
                    "Tag":      r["Tag"],
                    "Metric":   "Recall",
                    "Value":    round(float(r["Recall"]), round_to),
                })

            # IOB2 (strict)
            iob2 = self.normalize_spans(content.entity_strict_report.copy())
            iob2 = iob2[~iob2["Tag"].isin(["micro", "macro", "weighted"])]
            for _, r in iob2.iterrows():
                rows.append({
                    "Language": lang_lbl,
                    "Scheme":   "IOB2",
                    "Tag":      r["Tag"],
                    "Metric":   "Precision",
                    "Value":    round(float(r["Precision"]), round_to),
                })
                rows.append({
                    "Language": lang_lbl,
                    "Scheme":   "IOB2",
                    "Tag":      r["Tag"],
                    "Metric":   "Recall",
                    "Value":    round(float(r["Recall"]), round_to),
                })

        return pd.DataFrame(rows, columns=["Language","Scheme","Tag","Metric","Value"])


class EntitySpanSupportHelper(BaseDashDataProcessor):
    """
    Extract Support counts per entity type from both IOB1 and IOB2 reports.
    Output columns: Language, Scheme, Tag, Support
    """

    def generate_df(self, selected_variant: str, round_to: int = 0) -> pd.DataFrame:
        rows = []

        for vkey in self._resolve_keys_for_variant(selected_variant):
            if vkey not in self.dash_data:
                continue
            lang_lbl = self.ds_label(vkey)
            content  = self.dash_data[vkey]

            # IOB1
            iob1 = self.normalize_spans(content.entity_non_strict_report.copy())
            iob1 = iob1[~iob1["Tag"].isin(["micro", "macro", "weighted"])]
            for _, r in iob1.iterrows():
                rows.append({
                    "Language": lang_lbl,
                    "Scheme":   "IOB1",
                    "Tag":      r["Tag"],
                    "Support":  round(float(r["Support"]), round_to),
                })

            # IOB2
            iob2 = self.normalize_spans(content.entity_strict_report.copy())
            iob2 = iob2[~iob2["Tag"].isin(["micro", "macro", "weighted"])]
            for _, r in iob2.iterrows():
                rows.append({
                    "Language": lang_lbl,
                    "Scheme":   "IOB2",
                    "Tag":      r["Tag"],
                    "Support":  round(float(r["Support"]), round_to),
                })

        return pd.DataFrame(rows, columns=["Language","Scheme","Tag","Support"])


class EntitySpanPredictionOutcomeBreakdownHelper(BaseDashDataProcessor):
    """
    Build tidy rows with TP/FP/FN per entity, for IOB1 & IOB2:
      Language, Scheme, Tag, Metric (TP|FP|FN), Scale (proportion), Count (raw)
    Expects on each DashData:
      - entity_non_strict_confusion_data['confusion_matrix']
      - entity_strict_confusion_data['confusion_matrix']
      Each confusion_matrix is a dict-like with TP/FP/FN per tag (rows).
    """

    def _prepare_one(self, cm: dict, scheme_label: str, lang_lbl: str) -> list[dict]:
        # cm -> DataFrame (rows=Tag, cols=TP/FP/FN) — your input was already like that
        df = pd.DataFrame(cm).T.reset_index().rename(columns={"index": "Tag"})
        df = self.normalize_spans(df)                                # PERS -> PER
        df = df[~df["Tag"].isin(["micro", "macro", "weighted"])]     # safety

        # ensure numeric
        for c in ("TP", "FP", "FN"):
            if c in df.columns:
                print(df)
                print(df[c])
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)
                print(df[c])
            else:
                df[c] = 0.0

        # per-entity total (TP+FP+FN)
        totals = (df["TP"] + df["FP"] + df["FN"]).replace(0, np.nan)

        rows = []
        for metric in ("TP", "FP", "FN"):
            scale = df[metric] / totals
            for tag, sc, cnt in zip(df["Tag"], scale.fillna(0.0), df[metric]):
                rows.append({
                    "Language": lang_lbl,
                    "Scheme": scheme_label,
                    "Tag": tag,
                    "Metric": metric,   # TP / FP / FN
                    "Scale": float(sc), # proportion within this tag
                    "Count": int(round(float(cnt))),
                })
        return rows

    def generate_df(self, selected_variant: str) -> pd.DataFrame:
        out = []
        for vkey in self._resolve_keys_for_variant(selected_variant):
            if vkey not in self.dash_data:
                continue
            lang_lbl = self.ds_label(vkey)
            content  = self.dash_data[vkey]

            # IOB1
            if getattr(content, "entity_non_strict_confusion_data", None):
                cm1 = content.entity_non_strict_confusion_data.get("confusion_matrix", {})
                out.extend(self._prepare_one(cm1, "IOB1", lang_lbl))

            # IOB2
            if getattr(content, "entity_strict_confusion_data", None):
                cm2 = content.entity_strict_confusion_data.get("confusion_matrix", {})
                out.extend(self._prepare_one(cm2, "IOB2", lang_lbl))

        return pd.DataFrame(out, columns=["Language","Scheme","Tag","Metric","Scale","Count"])


# class EntitySpanConfusionHelper(BaseDashDataProcessor):
#     """
#     Build a tidy frame of confusion components per entity span from both schemes.
#     Output columns:
#       Language, Scheme, Tag, Metric, Count, Scale
#     Where:
#       Metric ∈ {"TP","FP","FN"}
#       Count = raw integer counts
#       Scale = Count / (TP+FP+FN) per (Language, Scheme, Tag)
#     """

#     @staticmethod
#     def _normalize_spans(df: pd.DataFrame) -> pd.DataFrame:
#         out = df.copy()
#         if "Tag" in out.columns:
#             out["Tag"] = out["Tag"].astype(str).replace(_SPAN_NORMALIZE)
#         return out

#     def _one_scheme(self, content, scheme_name: str) -> pd.DataFrame:
#         """
#         content.entity_non_strict_confusion_data['confusion_matrix']  (IOB1)
#         content.entity_strict_confusion_data['confusion_matrix']      (IOB2)
#         Expected shape: dict of rows keyed by Tag with TP/FP/FN
#         """
#         if scheme_name == "IOB1":
#             raw = getattr(content, "entity_non_strict_confusion_data", None)
#         else:
#             raw = getattr(content, "entity_strict_confusion_data", None)
#         if not raw or "confusion_matrix" not in raw:
#             return pd.DataFrame(columns=["Tag","TP","FP","FN"])

#         mat = pd.DataFrame(raw["confusion_matrix"]).T.reset_index().rename(columns={"index":"Tag"})
#         return self._normalize_spans(mat)

#     def generate_df(self, selected_variant: str) -> pd.DataFrame:
#         """
#         Returns tidy:
#           Language, Scheme, Tag, Metric, Count, Scale
#         (Scale is per Tag within Language×Scheme; we exclude 'O' if present)
#         """
#         # we DO NOT call build_corpora; we need raw dash_data reports
#         variant_keys = self._resolve_keys_for_variant(selected_variant)
#         rows: List[pd.DataFrame] = []

#         for vkey in variant_keys:
#             if vkey not in self.dash_data:
#                 continue
#             content = self.dash_data[vkey]
#             lang_lbl = self.ds_label(vkey)  # your Language (Arabic/English or model label)

#             # Both schemes
#             for scheme in ("IOB1", "IOB2"):
#                 mat = self._one_scheme(content, scheme)
#                 if mat.empty:
#                     continue

#                 # Optional: drop 'O' if it sneaks in
#                 mat = mat[~mat["Tag"].astype(str).str.upper().eq("O")].copy()

#                 # melt to tidy metrics
#                 melted = mat.melt(id_vars=["Tag"], value_vars=["TP","FP","FN"],
#                                   var_name="Metric", value_name="Count")
#                 # total per tag to create Scale
#                 denom = (
#                     melted.groupby("Tag", as_index=False)["Count"]
#                           .sum()
#                           .rename(columns={"Count":"Total"})
#                 )
#                 out = melted.merge(denom, on="Tag", how="left")
#                 out["Scale"] = np.where(out["Total"] > 0, out["Count"] / out["Total"], 0.0)

#                 out["Language"] = lang_lbl
#                 out["Scheme"]   = scheme

#                 rows.append(out[["Language","Scheme","Tag","Metric","Count","Scale"]])

#         return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
#             columns=["Language","Scheme","Tag","Metric","Count","Scale"]
#         )

class EntitySpanConfusionHelper(BaseDashDataProcessor):
    """
    Build a tidy DF of FP/FN counts per Tag for both schemes.
    Columns: Language, Model, Scheme, Tag, Metric, Count, Share
    """
    def generate_df(self, selected_variant: str) -> pd.DataFrame:
        # self.build_corpora(selected_variant)
        rows = []

        for ds_key, content in self.dash_data.items():
            # language label (Arabic/English, etc.)
            lang = self.ds_label(ds_key)
            # model label (AraBERTv02/BERT, etc.)

            # ---- Non-strict (IOB1)
            m1 = pd.DataFrame(content.entity_non_strict_confusion_data["confusion_matrix"]).T
            m1["Scheme"] = "IOB1"
            m1["Language"] = lang

            # ---- Strict (IOB2)
            m2 = pd.DataFrame(content.entity_strict_confusion_data["confusion_matrix"]).T
            m2["Scheme"] = "IOB2"
            m2["Language"] = lang

            mm = pd.concat([m1, m2], axis=0)
            mm = mm.rename_axis("Tag").reset_index()

            # normalize tags (PER vs PERS) if you want
            mm = self.normalize_spans(mm)  # uses your Base helper
            # melt FP/FN to long form
            long = mm.melt(
                id_vars=["Language", "Scheme", "Tag"],
                value_vars=["FP", "FN"],
                var_name="Metric",
                value_name="Count",
            )
            # compute per (Language, Model, Scheme, Tag) totals for shares
            totals = long.groupby(["Language", "Scheme", "Tag"], as_index=False)["Count"].sum()
            totals = totals.rename(columns={"Count": "Total"})
            long = long.merge(totals, on=["Language", "Scheme", "Tag"], how="left")
            long["Share"] = (long["Count"] / long["Total"]).fillna(0.0)

            rows.append(long[["Language",  "Scheme", "Tag", "Metric", "Count", "Share"]])

        return pd.concat(rows, ignore_index=True)


# class SpanErrorTypesHelper(BaseDashDataProcessor):
#     """
#     Build tidy error-type distribution for span-level FP/FN breakdown.
#     Output columns: Language, Scheme, Component (FP/FN), Error Type, Count, Percentage
#     """

#     def generate_df(self, selected_variant: str, round_to: int = 2) -> pd.DataFrame:
#         self.build_corpora(selected_variant)
#         rows = []
#         error_components = ["false_positives", "false_negatives"]

#         for component in error_components:
#             o_error = "Inclusion" if component == "false_positives" else "Exclusion"

#             for ds_key, content in self.dash_data.items():
#                 lang_lbl  = self.ds_label(ds_key)

#                 # we only keep IOB2 for now, extend if you want both
#                 for scheme, entity_confusion in [
#                     ("IOB1", content.entity_non_strict_confusion_data),
#                     ("IOB2", content.entity_strict_confusion_data),
#                 ]:
#                     # process confusion into entity/boundary breakdown
#                     error_types, _ = self.process_entity_confusion(entity_confusion[component], o_error)

#                     # rename O category
#                     error_types.rename(columns={o_error: "O Errors"}, inplace=True)

#                     # melt tidy
#                     melted = error_types.melt(
#                         var_name="Error Type",
#                         value_name="Count"
#                     )
#                     melted["Language"]  = lang_lbl
#                     melted["Scheme"]    = scheme
#                     melted["Component"] = "False Positives" if component == "false_positives" else "False Negatives"

#                     rows.append(melted)

#         df = pd.concat(rows, ignore_index=True)

#         # compute percentages within each (Language, Model, Scheme, Component)
#         df["Percentage"] = (
#             df.groupby(["Language",  "Scheme", "Component"])["Count"]
#               .transform(lambda x: (x / x.sum()) * 100)
#               .round(round_to)
#         )

#         return df


class SpanErrorTypesHelper(BaseDashDataProcessor):
    """
    Build tidy error-type distribution for span-level FP/FN breakdown.
    Output columns: Language, Scheme, Component (FP/FN), Error Type, Count, Percentage
    """

    def generate_df(self, selected_variant: str, round_to: int = 2) -> pd.DataFrame:
        # self.build_corpora(selected_variant)
        rows = []
        error_components = ["false_positives", "false_negatives"]

        for component in error_components:
            o_error = "Inclusion" if component == "false_positives" else "Exclusion"

            for ds_key, content in self.dash_data.items():
                lang_lbl = self.ds_label(ds_key)

                # include both schemes
                for scheme, entity_confusion in [
                    ("IOB1", getattr(content, "entity_non_strict_confusion_data", {})),
                    ("IOB2", getattr(content, "entity_strict_confusion_data", {})),
                ]:
                    if not entity_confusion or component not in entity_confusion:
                        continue

                    # process confusion into high-level categories (rows = tags)
                    error_types, _ = self.process_entity_confusion(entity_confusion[component], o_error)

                    # rename O-category
                    error_types = error_types.rename(columns={o_error: "O Errors"})

                    # ---- Option A (recommended): aggregate across tags ----
                    totals = (
                        error_types.sum(axis=0)              # Series indexed by error-type
                                   .rename("Count")          # name the value column
                                   .reset_index()            # -> cols: ['index','Count']
                                   .rename(columns={"index": "Error Type"})
                    )
                    totals["Language"]  = lang_lbl
                    totals["Scheme"]    = scheme
                    totals["Component"] = "False Positives" if component == "false_positives" else "False Negatives"
                    rows.append(totals)

                    # ---- Option B (per-tag detail) ----
                    # per_tag = (
                    #     error_types.reset_index()
                    #                .rename(columns={"index": "Tag"})
                    #                .melt(id_vars="Tag", var_name="Error Type", value_name="Count")
                    # )
                    # per_tag["Language"]  = lang_lbl
                    # per_tag["Scheme"]    = scheme
                    # per_tag["Component"] = "False Positives" if component == "false_positives" else "False Negatives"
                    # rows.append(per_tag)

        if not rows:
            return pd.DataFrame(columns=["Language","Scheme","Component","Error Type","Count","Percentage"])

        df = pd.concat(rows, ignore_index=True)

        # Percentages within each (Language, Scheme, Component)
        df["Percentage"] = (
            df.groupby(["Language", "Scheme", "Component"])["Count"]
              .transform(lambda x: (x / x.sum()) * 100)
              .round(round_to)
        )

        # Nice consistent order for the bars
        err_order = ["Entity", "Boundary", "Entity and Boundary", "O Errors"]
        comp_order = ["False Positives", "False Negatives"]
        df["Error Type"] = pd.Categorical(df["Error Type"], categories=err_order, ordered=True)
        df["Component"]  = pd.Categorical(df["Component"],  categories=comp_order, ordered=True)

        return df



class SpanErrorTypesHeatmapHelper(BaseDashDataProcessor):
    """
    Tidy DF for error-type heatmaps.
    Output columns:
      Language | Scheme | Component (FP/FN) | Tag (Entity) | Metric (Error Type) | Count
    """

    def generate_df(self, selected_variant: str, component: str) -> pd.DataFrame:
        """
        component ∈ {"false_positives", "false_negatives"}
        """
        assert component in {"false_positives", "false_negatives"}
        # self.build_corpora(selected_variant)

        rows = []
        o_error = "Inclusion" if component == "false_positives" else "Exclusion"
        comp_lbl = "False Positives" if component == "false_positives" else "False Negatives"

        for ds_key, content in self.dash_data.items():
            lang_lbl = self.ds_label(ds_key)

            for scheme, entity_confusion in [
                ("IOB1", getattr(content, "entity_non_strict_confusion_data", {})),
                ("IOB2", getattr(content, "entity_strict_confusion_data", {})),
            ]:
                if not entity_confusion or component not in entity_confusion:
                    continue

                # Process into error categories
                error_types, _ = self.process_entity_confusion(entity_confusion[component], o_error)
                error_types = error_types.rename(columns={o_error: "O Errors"})

                # index = entity tag (LOC, ORG, …), cols = error types
                error_types = error_types.reset_index().rename(columns={"index": "Tag"})
                error_types = self.normalize_spans(error_types, col="Tag")  # ✅ normalize tags (PER vs PERS)


                # Long format: one row per Tag × Error Type
                melted = error_types.melt(
                    id_vars="Tag",
                    var_name="Metric",    # error type (Entity, Boundary, etc.)
                    value_name="Count"
                )
                melted["Language"]  = lang_lbl
                melted["Scheme"]    = scheme
                melted["Component"] = comp_lbl

                rows.append(melted)

        if not rows:
            return pd.DataFrame(columns=["Language", "Scheme", "Component", "Tag", "Metric", "Count"])

        df = pd.concat(rows, ignore_index=True)

        # enforce consistent error type order
        metric_order = ["Entity", "Boundary", "Entity and Boundary", "O Errors"]
        df["Metric"] = pd.Categorical(df["Metric"], categories=metric_order, ordered=True)

        return df


class SpanEntityErrorsHeatmapHelper(BaseDashDataProcessor):
    """
    Build tidy DF for Predicted-vs-True *entity* errors (no Boundary/O).
    Output columns: Language, Scheme, True, Pred, Count
    component ∈ {"false_positives", "false_negatives"}
    """

    def generate_df(self, selected_variant: str, component: str) -> pd.DataFrame:
        assert component in {"false_positives", "false_negatives"}
        # self.build_corpora(selected_variant)

        rows = []
        o_error = "Inclusion" if component == "false_positives" else "Exclusion"

        for ds_key, content in self.dash_data.items():
            lang_lbl = self.ds_label(ds_key)

            for scheme, conf in [
                ("IOB1", getattr(content, "entity_non_strict_confusion_data", {})),
                ("IOB2", getattr(content, "entity_strict_confusion_data", {})),
            ]:
                if not conf or component not in conf:
                    continue

                # Base transform you defined earlier
                _, entity_errors = self.process_entity_confusion(conf[component], o_error)
                # entity_errors: rows = True entity, cols = Predicted entity (PER/ORG/LOC/MISC...)
                # drop any non-entity cols if present (safety)
                drop_cols = [c for c in ["Boundary", "Entity and Boundary", "O", o_error] if c in entity_errors.columns]
                if drop_cols:
                    entity_errors = entity_errors.drop(columns=drop_cols, errors="ignore")

                # index is True entity → make it a column and normalise spans on both axes
                df = entity_errors.reset_index().rename(columns={"index": "True"})
                df = self.normalize_spans(df, col="True")
                # melt predicted entities
                df = df.melt(id_vars=["True"], var_name="Pred", value_name="Count")
                df = self.normalize_spans(df, col="Pred")

                df["Language"] = lang_lbl
                df["Scheme"]   = scheme
                rows.append(df)

        if not rows:
            return pd.DataFrame(columns=["Language", "Scheme", "True", "Pred", "Count"])

        out = pd.concat(rows, ignore_index=True)
        # keep only nonzero counts for cleaner heatmaps (optional)
        # out = out[out["Count"] > 0]
        return out


class TokenF1Helper(BaseDashDataProcessor):
    """Token-level F1 per tag."""
    def generate_df(self, selected_variant: str, round_to: int = 3) -> pd.DataFrame:
        # self.build_corpora(selected_variant)
        rows = []
        for ds_key, content in self.dash_data.items():
            lang = self.ds_label(ds_key)
            tr = getattr(content, "token_report", None)
            if tr is None or tr.empty:
                continue
            df = tr.copy()
            if "Tag" in df.columns:
                df["Tag"] = df["Tag"].replace(_TAG_NORMALIZE)
                df = df[~df["Tag"].astype(str).str.lower().isin({t.lower() for t in EXCLUDE_ROWS})]
            for _, r in df.iterrows():
                rows.append({
                    "Language": lang,
                    "Tag": r["Tag"],
                    "F1-score": round(float(r.get("F1", r.get("F1-score", 0.0))), round_to)
                })
        return pd.DataFrame(rows)


class TokenSupportHelper(BaseDashDataProcessor):
    """Token-level support per tag."""
    def generate_df(self, selected_variant: str) -> pd.DataFrame:
        # self.build_corpora(selected_variant)
        rows = []
        for ds_key, content in self.dash_data.items():
            lang = self.ds_label(ds_key)
            tr = getattr(content, "token_report", None)
            if tr is None or tr.empty:
                continue
            df = tr.copy()
            if "Tag" in df.columns:
                df["Tag"] = df["Tag"].replace(_TAG_NORMALIZE)
                df = df[~df["Tag"].astype(str).str.lower().isin({t.lower() for t in EXCLUDE_ROWS})]
            for _, r in df.iterrows():
                rows.append({"Language": lang, "Tag": r["Tag"], "Support": int(r.get("Support", 0))})
        return pd.DataFrame(rows)

class TokenPrecisionRecallHelper(BaseDashDataProcessor):
    """Token-level Precision & Recall per tag (melted for grouped bars)."""
    def generate_df(self, selected_variant: str, round_to: int = 3) -> pd.DataFrame:
        # self.build_corpora(selected_variant)
        out = []
        for ds_key, content in self.dash_data.items():
            lang = self.ds_label(ds_key)
            tr = getattr(content, "token_report", None)
            if tr is None or tr.empty:
                continue
            df = tr.copy()
            if "Tag" in df.columns:
                df["Tag"] = df["Tag"].replace(_TAG_NORMALIZE)
                df = df[~df["Tag"].astype(str).str.lower().isin({t.lower() for t in EXCLUDE_ROWS})]
            m = df.melt(id_vars=["Tag"], value_vars=["Precision", "Recall"],
                        var_name="Metric", value_name="Score")
            m["Language"] = lang
            m["Score"] = pd.to_numeric(m["Score"], errors="coerce").round(round_to)
            out.append(m)
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
            columns=["Language","Tag","Metric","Score"]
        )


class TokenSupportHelper(BaseDashDataProcessor):
    """Token-level support per tag."""
    def generate_df(self, selected_variant: str) -> pd.DataFrame:
        # self.build_corpora(selected_variant) 
        rows = []
        for ds_key, content in self.dash_data.items():
            lang = self.ds_label(ds_key)
            tr = getattr(content, "token_report", None)
            if tr is None or tr.empty:
                continue
            df = tr.copy()
            if "Tag" in df.columns:
                df["Tag"] = df["Tag"].replace(_TAG_NORMALIZE)
                df = df[~df["Tag"].astype(str).str.lower().isin({t.lower() for t in EXCLUDE_ROWS})].copy()
                df = df[df["Tag"]!="O"]
            for _, r in df.iterrows():
                rows.append({"Language": lang, "Tag": r["Tag"], "Support": int(r.get("Support", 0))})
        return pd.DataFrame(rows)

class TokenPredictionOutcomesHelper(BaseDashDataProcessor):
    """
    Build tidy TP/FP/FN outcomes per token tag for each language.
    Output columns: Language, Tag, Metric (TP/FP/FN), Count, Share
    """

    def generate_df(self, selected_variant: str, round_to: int = 3) -> pd.DataFrame:
        rows = []
        for ds_key, content in self.dash_data.items():
            lang_lbl = self.ds_label(ds_key)

            cm = getattr(content, "token_confusion_matrix", None)
            if not cm or "confusion_matrix" not in cm:
                continue

            df = pd.DataFrame(cm["confusion_matrix"]).T.reset_index().rename(columns={"index": "Tag"})
            df = self.normalize_spans(df)  # 🔹 normalize PERS → PER
            df["Language"] = lang_lbl

            # total per tag
            df["Total"] = df[["TP", "FP", "FN"]].sum(axis=1)

            # preserve raw counts
            df["TP_Count"] = df["TP"]
            df["FP_Count"] = df["FP"]
            df["FN_Count"] = df["FN"]

            # shares (avoid div/0)
            for col in ["TP", "FP", "FN"]:
                df[col] = df[col] / df["Total"].replace(0, pd.NA)
            df['Tag'] = df['Tag'].replace(_TAG_NORMALIZE)
            # melt tidy
            scaled = df.melt(
                id_vars=["Language", "Tag"],
                value_vars=["TP", "FP", "FN"],
                var_name="Metric",
                value_name="Scaled Count"
            )
            
            counts = df.melt(
                id_vars=["Language", "Tag"],
                value_vars=["TP_Count", "FP_Count", "FN_Count"],
                var_name="Metric",
                value_name="Count"
            )
            counts["Metric"] = counts["Metric"].str.replace("_Count", "", regex=False)

            out = scaled.merge(counts, on=["Language", "Tag", "Metric"], how="left")
            out["Scaled Count"] = out["Scaled Count"].round(round_to)

            rows.append(out)

        if not rows:
            return pd.DataFrame(columns=["Language", "Tag", "Metric", "Scaled Count", "Count"])

        return pd.concat(rows, ignore_index=True)


class TokenMisclassHeatmapHelper(BaseDashDataProcessor):
    """
    Tidy DF for token-level misclassification heatmap.
    Output: Language, True, Pred, Count
    """

    def _normalize_index_and_cols(self, m: pd.DataFrame) -> pd.DataFrame:
        # Reuse your span normalizer: B-PERS/I-PERS → B-PER/I-PER, etc.
        # If you already have normalize_spans for Series, just apply to both axes.
        return m.rename(index=_TAG_NORMALIZE, columns=_TAG_NORMALIZE)

    def generate_df(self, selected_variant: str) -> pd.DataFrame:
        rows = []
        # (No need to build_corpora; we just read dash_data)
        for ds_key, content in self.dash_data.items():
            lang = self.ds_label(ds_key)

            mat = pd.DataFrame(getattr(content, "token_misclassifications"))
            if mat is None or mat.empty:
                continue

            # normalize both axes
            mat = self._normalize_index_and_cols(mat)

            # tidy melt: True x Pred → Count
            tidy = (
                mat.reset_index()
                   .rename(columns={"index": "True"})
                   .melt(id_vars="True", var_name="Pred", value_name="Count")
            )
            tidy["Language"] = lang
            rows.append(tidy)

        if not rows:
            return pd.DataFrame(columns=["Language", "True", "Pred", "Count"])

        return pd.concat(rows, ignore_index=True)


class TokenSupportCorrelationHelper(BaseDashDataProcessor):
    """
    Computes Pearson & Spearman correlations between (Train Support, Test Support)
    and (Precision, Recall) per model/language.

    Output (tidy):
      Language | Split(train/test) | Method(pearson/spearman) | Metric(Precision/Recall) | Corr
    """

    EXCLUDE = {"O", "micro", "macro", "weighted"}

    def _make_report_df(self) -> pd.DataFrame:
        rows = []
        for ds_key, content in self.dash_data.items():
            lang = self.ds_label(ds_key)

            tok = getattr(content, "token_report", None)
            tr  = getattr(content, "train_data", None)
            if tok is None or tok.empty or tr is None or tr.empty:
                continue

            # --- test support from token_report ---
            df = tok.copy()
            if "Tag" in df.columns:
                df["Tag"] = df["Tag"].replace(_TAG_NORMALIZE)
                df = df[~df["Tag"].astype(str).str.upper().isin({t.upper() for t in self.EXCLUDE})]

            # --- train support from train_data (ignore -100) ---
            tr_df = tr[tr["Labels"] != -100]
            train_support = (tr_df["True Labels"]
                             .replace(_TAG_NORMALIZE)
                             .value_counts()
                             .rename_axis("Tag")
                             .reset_index(name="Train Support"))

            # merge & annotate
            merged = df.merge(train_support, on="Tag", how="left")
            merged["Language"] = lang
            rows.append(merged)

        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def generate_df(self, round_to: int = 3) -> pd.DataFrame:
        data = self._make_report_df()
        if data.empty:
            return pd.DataFrame(columns=["Language","Split","Method","Metric","Corr"])

        # Rename test support for clarity
        data = data.rename(columns={"Support": "Test Support"})

        out = []
        for lang, g in data.groupby("Language"):
            for metric in ["Precision", "Recall"]:
                # Pearson
                pearson_train = g[["Train Support", metric]].corr().iloc[0, 1]
                pearson_test  = g[["Test Support",  metric]].corr().iloc[0, 1]
                out += [
                    {"Language": lang, "Split": "Train", "Method": "Pearson",  "Metric": metric, "Corr": pearson_train},
                    {"Language": lang, "Split": "Test",  "Method": "Pearson",  "Metric": metric, "Corr": pearson_test},
                ]
                # Spearman
                spear_train = g[["Train Support", metric]].corr(method="spearman").iloc[0, 1]
                spear_test  = g[["Test Support",  metric]].corr(method="spearman").iloc[0, 1]
                out += [
                    {"Language": lang, "Split": "Train", "Method": "Spearman", "Metric": metric, "Corr": spear_train},
                    {"Language": lang, "Split": "Test",  "Method": "Spearman", "Metric": metric, "Corr": spear_test},
                ]

        df = pd.DataFrame(out)
        df["Corr"] = pd.to_numeric(df["Corr"], errors="coerce").round(round_to)
        return df

class TokenSupportScatterHelper(BaseDashDataProcessor):
    """
    Build the tidy data for: Support (Train/Test) vs Precision/Recall, per tag.
    Returns two DFs:
      - points_df: one row per (Language, Tag, Split, Metric) with Value and Support Value
      - means_df: facet means/stats per (Language, Metric, Split)
    """

    EXCLUDE_TAGS = {"O"}

    def _token_train_support(self, tr_df: pd.DataFrame) -> pd.DataFrame:
        sub = tr_df[tr_df["Labels"] != -100]
        vc = sub["True Labels"].value_counts().reset_index()
        vc.columns = ["Tag", "Train Support"]
        return vc

    def generate_df(
        self,
        selected_variant: str,
        *,
        language: str | None = None,
        round_to: int = 3
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """If `language` is provided, return frames already filtered to that language."""
        points = []

        # If you want to scope to a specific variant/model, uncomment the next two lines:
        # variant_keys = self._resolve_keys_for_variant(selected_variant)
        # dash_items = {k: v for k, v in self.dash_data.items() if k in variant_keys}
        dash_items = self.dash_data  # current behavior: use everything present

        for ds_key, content in dash_items.items():
            lang = self.ds_label(ds_key)  # e.g. "Arabic", "English"
            if language is not None and lang != language:
                continue

            token_report = getattr(content, "token_report", None)
            train_df     = getattr(content, "train_data", None)
            if token_report is None or token_report.empty or train_df is None:
                continue

            rep = token_report.copy()
            if "Tag" in rep.columns:
                rep["Tag"] = rep["Tag"].replace(_TAG_NORMALIZE)
                rep = rep[~rep["Tag"].astype(str).isin(self.EXCLUDE_TAGS)]

            tr_support = self._token_train_support(train_df)
            rep = rep.merge(tr_support, on="Tag", how="left")

            melt = rep.melt(
                id_vars=["Tag", "Support", "Train Support"],
                value_vars=["Precision", "Recall"],
                var_name="Metric",
                value_name="Value",
            )

            m_train = melt.copy()
            m_train["Split"] = "Train"
            m_train["Support Value"] = m_train["Train Support"]

            m_test = melt.copy()
            m_test["Split"] = "Test"
            m_test["Support Value"] = m_test["Support"]

            both = pd.concat([m_train, m_test], ignore_index=True)
            both["Language"] = lang
            both["Value"] = pd.to_numeric(both["Value"], errors="coerce").round(round_to)
            both["Support Value"] = pd.to_numeric(both["Support Value"], errors="coerce")

            points.append(both)

        if not points:
            return (
                pd.DataFrame(columns=["Language","Tag","Split","Metric","Value","Support Value"]),
                pd.DataFrame(columns=[
                    "Language","Metric","Split","Mean_Support","Mean_Metric",
                    "Std_Support","Std_Metric","Max_Support","Min_Support",
                    "Max_Metric","Min_Metric","Support_Spread"
                ]),
            )

        points_df = pd.concat(points, ignore_index=True)

        means_df = (
            points_df
            .groupby(["Language","Metric","Split"], observed=False)
            .agg(
                Mean_Support=("Support Value","mean"),
                Mean_Metric =("Value","mean"),
                Std_Support =("Support Value","std"),
                Std_Metric  =("Value","std"),
                Max_Support =("Support Value","max"),
                Min_Support =("Support Value","min"),
                Max_Metric  =("Value","max"),
                Min_Metric  =("Value","min"),
            )
            .reset_index()
        )
        means_df["Support_Spread"] = means_df["Max_Support"] - means_df["Min_Support"]

        # already filtered above if language != None
        return points_df, means_df

    # convenience wrapper: *only* one language returned
    def generate_for_language(
        self,
        selected_variant: str,
        language: str,
        round_to: int = 3,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.generate_df(selected_variant, language=language, round_to=round_to)


class TokenSpearmanHelper(BaseDashDataProcessor):
    """
    Spearman rank contributions for Precision/Recall vs Support (Train/Test).
    Output: Tag | Language | Metric | Split | X Rank | Y Rank | Rank Difference
            | Higher Column | Squared Rank Difference
    """

    EXCLUDE = {"O", "micro", "macro", "weighted"}

    def generate_df(
        self,
        selected_variant: str,
        *,
        language: str | None = None,
        round_to: int = 3
    ) -> pd.DataFrame:
        rows = []

        # variant scoping if needed:
        # variant_keys = self._resolve_keys_for_variant(selected_variant)
        # dash_items = {k: v for k, v in self.dash_data.items() if k in variant_keys}
        dash_items = self.dash_data

        for ds_key, content in dash_items.items():
            lang = self.ds_label(ds_key)
            if language is not None and lang != language:
                continue

            tok = getattr(content, "token_report", None)
            tr  = getattr(content, "train_data", None)
            if tok is None or tok.empty:
                continue

            rep = tok.copy()
            if "Tag" in rep.columns:
                rep["Tag"] = rep["Tag"].replace(_TAG_NORMALIZE)
                rep = rep[~rep["Tag"].astype(str).str.upper().isin({t.upper() for t in self.EXCLUDE})]

            if tr is not None and not tr.empty:
                tr_df = tr[tr["Labels"] != -100]
                tr_support = (
                    tr_df["True Labels"]
                    .replace(_TAG_NORMALIZE)
                    .value_counts()
                    .rename_axis("Tag")
                    .reset_index(name="Train Support")
                )
                rep = rep.merge(tr_support, on="Tag", how="left")
            else:
                rep["Train Support"] = np.nan

            rep["Language"] = lang

            for split, sup_col in [("Train", "Train Support"), ("Test", "Support")]:
                for metric in ["Precision", "Recall"]:
                    sub = rep[["Tag", sup_col, metric]].dropna()
                    if sub.empty:
                        continue

                    sub[sup_col] = pd.to_numeric(sub[sup_col], errors="coerce")
                    sub[metric]  = pd.to_numeric(sub[metric], errors="coerce")
                    sub = sub.dropna()
                    if sub.empty:
                        continue

                    sub["X Rank"] = rankdata(sub[sup_col].to_numpy(), method="average")
                    sub["Y Rank"] = rankdata(sub[metric].to_numpy(), method="average")
                    sub["Rank Difference"] = sub["X Rank"] - sub["Y Rank"]
                    sub["Squared Rank Difference"] = sub["Rank Difference"] ** 2
                    sub["Higher Column"] = np.where(
                        sub["X Rank"] > sub["Y Rank"], "Support Higher", "Metric Higher"
                    )
                    sub["Split"] = split
                    sub["Metric"] = metric
                    sub["Language"] = lang

                    rows.append(sub[[
                        "Tag","Language","Metric","Split",
                        "X Rank","Y Rank","Rank Difference",
                        "Higher Column","Squared Rank Difference"
                    ]])

        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    # convenience wrapper: *only* one language returned
    def generate_for_language(self, selected_variant: str, language: str, round_to: int = 3) -> pd.DataFrame:
        return self.generate_df(selected_variant, language=language, round_to=round_to)

# class TokenSupportScatterHelper(BaseDashDataProcessor):
#     """
#     Build the tidy data for: Support (Train/Test) vs Precision/Recall, per tag.
#     Returns two DFs:
#       - points_df: one row per (Language, Tag, Split, Metric) with Value and Support Value
#       - means_df: facet means/stats per (Language, Metric, Split)
#     """

#     EXCLUDE_TAGS = {"O"}

#     def _token_train_support(self, tr_df: pd.DataFrame) -> pd.DataFrame:
#         """Compute Train Support per tag from train_data (labels != -100)."""
#         sub = tr_df[tr_df["Labels"] != -100]
#         vc = sub["True Labels"].value_counts().reset_index()
#         vc.columns = ["Tag", "Train Support"]
#         return vc

#     def generate_df(self, selected_variant: str, round_to: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
#         # No need to build corpora here; use existing dash_data
#         points = []

#         for ds_key, content in self.dash_data.items():
#             lang = self.ds_label(ds_key)  # e.g., "Arabic", "English"

#             token_report = getattr(content, "token_report", None)
#             train_df     = getattr(content, "train_data", None)
#             if token_report is None or token_report.empty or train_df is None:
#                 continue

#             # Normalize tags & drop unwanted rows
#             rep = token_report.copy()
#             if "Tag" in rep.columns:
#                 rep["Tag"] = rep["Tag"].replace(_TAG_NORMALIZE)
#                 rep = rep[~rep["Tag"].astype(str).isin(self.EXCLUDE_TAGS)]

#             # Merge Train Support
#             tr_support = self._token_train_support(train_df)
#             rep = rep.merge(tr_support, on="Tag", how="left")

#             # Build melted PR (Precision/Recall) long-form
#             melt = rep.melt(
#                 id_vars=["Tag", "Support", "Train Support"],
#                 value_vars=["Precision", "Recall"],
#                 var_name="Metric",
#                 value_name="Value",
#             )

#             # Split Train/Test
#             m_train = melt.copy()
#             m_train["Split"] = "Train"
#             m_train["Support Value"] = m_train["Train Support"]

#             m_test = melt.copy()
#             m_test["Split"] = "Test"
#             m_test["Support Value"] = m_test["Support"]

#             both = pd.concat([m_train, m_test], ignore_index=True)
#             both["Language"] = lang

#             # Nice numeric cleanup
#             both["Value"] = pd.to_numeric(both["Value"], errors="coerce").round(round_to)
#             both["Support Value"] = pd.to_numeric(both["Support Value"], errors="coerce")

#             points.append(both)

#         if not points:
#             # empty fallbacks
#             return (
#                 pd.DataFrame(columns=["Language","Tag","Split","Metric","Value","Support Value"]),
#                 pd.DataFrame(columns=["Language","Metric","Split","Mean_Support","Mean_Metric","Std_Support","Std_Metric","Support_Spread"])
#             )

#         points_df = pd.concat(points, ignore_index=True)

#         # Means per facet: (Language, Metric, Split)
#         means_df = (
#             points_df
#             .groupby(["Language","Metric","Split"], observed=False)
#             .agg(
#                 Mean_Support=("Support Value","mean"),
#                 Mean_Metric =("Value","mean"),
#                 Std_Support =("Support Value","std"),
#                 Std_Metric  =("Value","std"),
#                 Max_Support =("Support Value","max"),
#                 Min_Support =("Support Value","min"),
#                 Max_Metric  =("Value","max"),
#                 Min_Metric  =("Value","min"),
#             )
#             .reset_index()
#         )
#         means_df["Support_Spread"] = means_df["Max_Support"] - means_df["Min_Support"]

#         return points_df, means_df


# class TokenSpearmanHelper(BaseDashDataProcessor):
#     """
#     Spearman rank contributions for Precision/Recall vs Support (Train/Test).
#     Output: Tag | Language | Metric | Split | X Rank | Y Rank | Rank Difference 
#             | Higher Column | Squared Rank Difference
#     """

#     EXCLUDE = {"O", "micro", "macro", "weighted"}

#     def generate_df(self, selected_variant: str, round_to: int = 3) -> pd.DataFrame:
#         rows = []

#         for ds_key, content in self.dash_data.items():
#             lang = self.ds_label(ds_key)
#             tok  = getattr(content, "token_report", None)
#             tr   = getattr(content, "train_data", None)

#             if tok is None or tok.empty:
#                 continue

#             rep = tok.copy()
#             # normalise & drop excluded
#             if "Tag" in rep.columns:
#                 rep["Tag"] = rep["Tag"].replace(_TAG_NORMALIZE)
#                 rep = rep[~rep["Tag"].astype(str).str.upper().isin({t.upper() for t in self.EXCLUDE})]

#             # attach train support
#             if tr is not None and not tr.empty:
#                 tr_df = tr[tr["Labels"] != -100]
#                 tr_support = (
#                     tr_df["True Labels"]
#                     .replace(_TAG_NORMALIZE)
#                     .value_counts()
#                     .rename_axis("Tag")
#                     .reset_index(name="Train Support")
#                 )
#                 rep = rep.merge(tr_support, on="Tag", how="left")
#             else:
#                 rep["Train Support"] = np.nan

#             rep["Language"] = lang

#             # per split × metric
#             for split, sup_col in [("Train", "Train Support"), ("Test", "Support")]:
#                 for metric in ["Precision", "Recall"]:
#                     sub = rep[["Tag", sup_col, metric]].dropna()
#                     if sub.empty:
#                         continue

#                     # numeric conversion
#                     sub[sup_col] = pd.to_numeric(sub[sup_col], errors="coerce")
#                     sub[metric]  = pd.to_numeric(sub[metric], errors="coerce")
#                     sub = sub.dropna()

#                     if sub.empty:
#                         continue

#                     # ranks: use average tie method, no zero-filling
#                     sub["X Rank"] = rankdata(sub[sup_col].to_numpy(), method="average")
#                     sub["Y Rank"] = rankdata(sub[metric].to_numpy(), method="average")

#                     sub["Rank Difference"] = sub["X Rank"] - sub["Y Rank"]
#                     sub["Squared Rank Difference"] = sub["Rank Difference"] ** 2
#                     sub["Higher Column"] = np.where(
#                         sub["X Rank"] > sub["Y Rank"], "Support Higher", "Metric Higher"
#                     )

#                     sub["Split"] = split
#                     sub["Metric"] = metric
#                     sub["Language"] = lang

#                     rows.append(
#                         sub[
#                             [
#                                 "Tag","Language","Metric","Split",
#                                 "X Rank","Y Rank","Rank Difference",
#                                 "Higher Column","Squared Rank Difference"
#                             ]
#                         ]
#                     )

#         return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
