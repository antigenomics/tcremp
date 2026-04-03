SUPPORTED_SINGLE_CHAINS = ("TRA", "TRB", "TRG", "TRD", "IGH", "IGK", "IGL")
DEFAULT_SINGLE_CHAIN_PROTOTYPE_RESOURCES = {
    "TRA": "tcremp_prototypes_TRA.tsv",
    "TRB": "tcremp_prototypes_TRB.tsv",
    "TRG": "tcremp_prototypes_TRG.tsv",
    "TRD": "tcremp_prototypes_TRD.tsv",
    "IGH": "tcremp_prototypes_IGH.tsv",
    "IGK": "tcremp_prototypes_IGK.tsv",
    "IGL": "tcremp_prototypes_IGL.tsv",
}
DEFAULT_PAIRED_CHAIN_COMPONENTS = {
    "TRA_TRB": ("TRA", "TRB"),
    "TRG_TRD": ("TRG", "TRD"),
    "IGH_IGL": ("IGH", "IGL"),
    "IGH_IGK": ("IGH", "IGK"),
}
SUPPORTED_CHAINS = SUPPORTED_SINGLE_CHAINS + tuple(DEFAULT_PAIRED_CHAIN_COMPONENTS)
DEFAULT_PROTOTYPE_CHAINS = set(SUPPORTED_CHAINS)
