from pathlib import Path
import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from tcremp import get_resource_path

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


colors = [
    "red", "cyan", "lime", "darkgreen", "gold", "pink", "lightsalmon",
    "yellow", "maroon", "blue", "teal", "orange", "olive", "indigo",
    "fuchsia", "palegreen", "crimson", "navy", "black",
]


def log_memory_usage(note=""):
    if psutil is None:
        return
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    logging.debug("[MEMORY] %s RSS memory usage: %.2f MB", note, mem_mb)


def configure_logging(input_path, output_path, output_prefix):
    formatter_str = "[%(asctime)s\t%(name)s\t%(levelname)s] %(message)s"
    formatter = logging.Formatter(formatter_str)
    logging.basicConfig(
        filename=f"{output_path}/{output_prefix}.log",
        format=formatter_str,
        level=logging.DEBUG,
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
    logging.info(
        'Running TCRemP for i="%s", writing to o="%s/" under prefix="%s"',
        input_path,
        Path(output_path).resolve(),
        output_prefix,
    )


def prepare_output_path(output):
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_input_file(file):
    return str(Path(file).resolve())


def resolve_prototype_file(path):
    if path:
        return str(Path(path).resolve())
    return str(Path(get_resource_path("tcremp_prototypes_olga.tsv")).resolve())


def generate_output_prefix(input_file, custom_prefix):
    return custom_prefix or Path(input_file).stem


def load_input_table(path):
    return pd.read_csv(path, sep=None, engine="python")


def get_label_metadata(input_path, index_col, labels_col):
    if labels_col is None:
        return None

    if index_col is None:
        logging.warning("labels-col was provided without index-col; skipping label transfer to t-SNE output.")
        return None

    df = load_input_table(input_path)
    missing = [col for col in (index_col, labels_col) if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s) in input table for metadata transfer: {missing}")

    meta = df[[index_col, labels_col]].drop_duplicates()
    duplicate_ids = meta[index_col].duplicated(keep=False)
    if duplicate_ids.any():
        logging.warning(
            "Found duplicate IDs for %s while transferring labels; keeping first label per ID.",
            index_col,
        )
        meta = meta.drop_duplicates(subset=[index_col], keep="first")

    return meta.rename(columns={index_col: "clone_id"})


def get_metadata_columns(input_path, index_col, columns):
    columns = [c for c in columns if c]
    if not columns:
        return None

    if index_col is None:
        logging.warning("Metadata columns were requested without index-col; skipping metadata transfer.")
        return None

    df = load_input_table(input_path)
    required = [index_col] + columns
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s) in input table for metadata transfer: {missing}")

    meta = df[required].drop_duplicates()
    duplicate_ids = meta[index_col].duplicated(keep=False)
    if duplicate_ids.any():
        logging.warning(
            "Found duplicate IDs for %s while transferring metadata; keeping first row per ID.",
            index_col,
        )
        meta = meta.drop_duplicates(subset=[index_col], keep="first")

    return meta.rename(columns={index_col: "clone_id"})


def validate_cdr3_len(repertoire, llen, hlen, single_chain):
    llen = llen if llen is not None else -1
    hlen = hlen if hlen is not None else 35

    if single_chain:
        predicate = lambda x: llen <= len(x.cdr3aa) < hlen
    else:
        predicate = (
            lambda x: llen <= len(x.chainA.cdr3aa) < hlen
            and llen <= len(x.chainB.cdr3aa) < hlen
        )

    return repertoire.subsample_by_lambda(predicate)


def _build_parser(segment_library, locus, mapping_column):
    from mir.common.parser import AIRRParser, DoubleChainAIRRParser

    if locus is not None:
        return AIRRParser(lib=segment_library, locus=locus)
    return DoubleChainAIRRParser(lib=segment_library, mapping_column=mapping_column)


def load_analysis_repertoire(path, segment_library, locus, mapping_column, llen, hlen):
    from mir.common.repertoire import Repertoire

    parser = _build_parser(segment_library, locus, mapping_column)
    repertoire = Repertoire.load(parser=parser, path=path)
    return validate_cdr3_len(repertoire, llen, hlen, single_chain=bool(locus))


def load_prototype_repertoire(path, segment_library, locus, mapping_column):
    from mir.common.repertoire import Repertoire

    parser = _build_parser(segment_library, locus, mapping_column)
    return Repertoire.load(parser=parser, path=path)


def subsample_repertoire(rep, n, sample_random, random_seed):
    if n and rep.total >= n:
        return rep.sample_n(n, sample_random=sample_random, random_seed=random_seed)
    return rep


def get_representations_df(rep, locus=None):
    df = pd.DataFrame({"clone_id": [c.id for c in rep]})

    def add_chain(clones, loc):
        df[f"cdr3aa_{loc}"] = [c.cdr3aa for c in clones]
        df[f"v_{loc}"] = [c.v.id for c in clones]
        df[f"j_{loc}"] = [c.j.id for c in clones]

    if locus is None:
        add_chain([x.chainA for x in rep], "alpha")
        add_chain([x.chainB for x in rep], "beta")
    else:
        add_chain(rep.clonotypes, locus)
    return df


def pca_proc(res_df, id_column="id", n_components=50):
    data = res_df.drop(id_column, axis=1, errors="ignore")
    n_components = min(n_components, data.shape[0], data.shape[1])
    if n_components < 1:
        raise ValueError("PCA requires at least one sample and one feature.")

    scaled = StandardScaler().fit_transform(data)
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(scaled)
    logging.info("PCA for visualization completed with %d components.", n_components)

    pca_data = pd.DataFrame(transformed, columns=[f"PC{x}" for x in range(n_components)])
    pca_data[id_column] = res_df[id_column].values
    return pca_data


def tsne_proc(proc_df, id_column="id", init="pca", random_state=7, perplexity=15):
    data = proc_df.drop(id_column, axis=1, errors="ignore")
    n_samples = len(data)
    if n_samples < 2:
        raise ValueError("t-SNE requires at least two samples.")

    perplexity = min(float(perplexity), max(1.0, float(n_samples - 1)))
    embedded = TSNE(
        n_components=2,
        init=init,
        random_state=random_state,
        perplexity=perplexity,
    ).fit_transform(data)
    tsne_data = pd.DataFrame(data=embedded, columns=["DM1", "DM2"])
    tsne_data[id_column] = proc_df[id_column].values
    logging.info("t-SNE for visualization completed with perplexity=%.2f.", perplexity)
    return tsne_data


def make_custom_palette(labels_list):
    palette = colors[:len(labels_list)]
    custom_palette = {labels_list[i]: palette[i] for i in range(len(labels_list))}
    custom_palette["other"] = "lightgrey"
    return custom_palette


def tsne_plot(data_plot, to_color, title, output_path, to_size=None, legend=True, custom_palette=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    if to_size is None:
        sns.scatterplot(
            x="DM1",
            y="DM2",
            data=data_plot.sort_values(to_color),
            hue=to_color,
            s=12,
            legend=legend,
            palette=custom_palette,
            ax=ax,
        )
    else:
        sns.scatterplot(
            x="DM1",
            y="DM2",
            data=data_plot.sort_values(to_color),
            hue=to_color,
            size=to_size,
            sizes=(12, 120),
            legend=legend,
            palette=custom_palette,
            ax=ax,
        )
    ax.set_title(title)
    if legend and ax.legend_ is not None:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
