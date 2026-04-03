import pandas as pd

from tcremp.arguments import get_arguments_enrich
from tcremp.enrichment import (
    annotate_clusters_with_enrichment,
    compute_cluster_enrichment,
    save_enrichment_outputs,
)
from tcremp.utils import configure_logging, generate_output_prefix, prepare_output_path, resolve_input_file


def main():
    args = get_arguments_enrich()
    input_path = resolve_input_file(args.input)
    output_dir = prepare_output_path(args.output)
    prefix = generate_output_prefix(args.input, args.prefix)
    configure_logging(input_path, output_dir, prefix)

    cluster_df = pd.read_csv(input_path, sep="\t")
    enrichment_summary = compute_cluster_enrichment(
        cluster_df=cluster_df,
        label_col=args.label_col,
        cluster_col=args.cluster_col,
        threshold=args.enrichment_threshold,
        fdr_threshold=args.enrichment_fdr_threshold,
    )
    annotated = annotate_clusters_with_enrichment(
        cluster_df=cluster_df,
        enrichment_summary=enrichment_summary,
        cluster_col=args.cluster_col,
    )
    save_enrichment_outputs(annotated, enrichment_summary, prefix, output_dir)


if __name__ == "__main__":
    main()
