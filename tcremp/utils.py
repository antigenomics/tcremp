from pathlib import Path
import logging
from mir.common.repertoire import Repertoire
from mir.common.parser import AIRRParser, DoubleChainAIRRParser
from tcremp import get_resource_path
import pandas as pd
from scipy.stats import fisher_exact
# from statsmodels.stats.multitest import multipletests
import psutil
import os


def log_memory_usage(note=""):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    logging.info(f"[MEMORY] {note} RSS memory usage: {mem_mb:.2f} MB")


def configure_logging(input_path, output_path, output_prefix):
    formatter_str = '[%(asctime)s\t%(name)s\t%(levelname)s] %(message)s'
    formatter = logging.Formatter(formatter_str)
    logging.basicConfig(filename=f'{output_path}/{output_prefix}.log',
                        format=formatter_str,
                        level=logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)


def prepare_output_path(output: str) -> Path:
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_input_file(file: str) -> str:
    return str(Path(file).resolve())


def resolve_prototype_file(path: str | None) -> str:
    return str(Path(path).resolve() if path else Path(get_resource_path('tcremp_prototypes_olga.tsv')).resolve())


def generate_output_prefix(input_file: str, custom_prefix: str | None) -> str:
    return custom_prefix or Path(input_file).stem


def resolve_embedding_file(custom_path: str | None, output_path: str, prefix: str, tag: str,
                           must_exist: bool = False) -> Path:
    """
    Resolves the path to an embedding file.

    Parameters:
        custom_path: User-provided path to embedding file (can be None).
        output_path: Base output directory.
        prefix: Output prefix (usually derived from input file).
        tag: Either "sample" or "background".
        must_exist: If True, raises an error if the resolved file does not exist.

    Returns:
        Path object pointing to the resolved file.
    """
    path = Path(custom_path) if custom_path else Path(output_path) / f"{prefix}_{tag}_embeddings.parquet"
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Embedding file for '{tag}' not found: {path}")
    return path


def validate_cdr3_len(repertoire, llen, hlen, single_chain):
    llen = llen if llen is not None else -1
    hlen = hlen if hlen is not None else 35
    if single_chain:
        f = lambda x: llen <= len(x.cdr3aa) < hlen
    else:
        f = lambda x: llen <= len(x.chainA.cdr3aa) < hlen and llen <= len(x.chainB.cdr3aa) < hlen
    return repertoire.subsample_by_lambda(f)


def load_analysis_repertoire(path, segment_library, locus, mapping_column, llen, hlen):
    parser = AIRRParser(lib=segment_library, locus=locus) if locus else \
             DoubleChainAIRRParser(lib=segment_library, mapping_column=mapping_column)
    rep = Repertoire.load(parser=parser, path=path)
    return validate_cdr3_len(rep, llen, hlen, single_chain=bool(locus))


def load_prototype_repertoire(path, segment_library, locus, mapping_column):
    parser = AIRRParser(lib=segment_library, locus=locus) if locus else \
             DoubleChainAIRRParser(lib=segment_library, mapping_column=mapping_column)
    return Repertoire.load(parser=parser, path=path)


def subsample_repertoire(rep, n, random, seed):
    if n and rep.total >= n:
        return rep.sample_n(n, sample_random=random, random_seed=seed)
    return rep


def get_representations_df(rep, locus=None):
    import pandas as pd
    df = pd.DataFrame({'clone_id': [c.id for c in rep]})

    def add_chain(clones, loc):
        df[f'cdr3aa_{loc}'] = [c.cdr3aa for c in clones]
        df[f'v_{loc}'] = [c.v.id for c in clones]
        df[f'j_{loc}'] = [c.j.id for c in clones]

    if locus is None:
        add_chain([x.chainA for x in rep], 'alpha')
        add_chain([x.chainB for x in rep], 'beta')
    else:
        add_chain(rep.clonotypes, locus)
    return df


def add_fisher_pvalues(summary: pd.DataFrame, total_sample: int, total_background: int) -> pd.DataFrame:
    pvals = []
    for _, row in summary.iterrows():
        a = row.get('sample', 0)
        c = row.get('background', 0)
        b = total_sample - a
        d = total_background - c

        contingency_table = [[a, b], [c, d]]
        _, pval = fisher_exact(contingency_table, alternative="greater")
        pvals.append(pval)

    summary['enrichment_pvalue'] = pvals
    # _, qvals, _, _ = multipletests(pvals, method="fdr_bh")
    # summary['enrichment_fdr'] = qvals
    return summary
