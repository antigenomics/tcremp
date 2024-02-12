from abc import abstractmethod
from itertools import starmap
from multiprocessing import Pool
from Bio import Align
from Bio.Align import substitution_matrices
import typing as t

from ..common import ClonotypeAA, Segment, SegmentLibrary


class Scoring:
    @abstractmethod
    def score(self, s1: str, s2: str) -> float:
        pass

    def score_norm(self, s1: str, s2: str) -> float:
        return self.score(s1, s2) - max(self.score(s1, s1), self.score(s2, s2))


class BioAlignerWrapper(Scoring):
    def __init__(self, scoring: str = "blastp"):
        self.aligner = Align.PairwiseAligner(scoring)

    def score(self, s1, s2) -> float:
        return self.aligner.align(s1, s2).score


# TODO substitution matrix wrapper to load from dict
class CDRAligner(Scoring):
    _factor = 10.

    def __init__(self,
                 gap_positions: t.Iterable[int] = (3, 4, -4, -3),
                 mat: substitution_matrices.Array = substitution_matrices.load(
                     'BLOSUM62'),
                 gap_penalty: float = -3,
                 v_offset: int = 3,
                 j_offset: int = 3):
        self.gap_positions = gap_positions
        self.mat = mat
        self.gap_penalty = gap_penalty
        self.v_offset = v_offset
        self.j_offset = j_offset

    def pad(self, s1, s2) -> tuple[tuple[str, str]]:
        d = len(s1) - len(s2)
        if d == 0:
            return tuple([(s1, s2)])
        elif d < 0:
            return tuple((s1[:p] + ('-' * d) + s1[p:], s2) for p in self.gap_positions)
        else:
            return tuple((s1, s2[:p] + ('-' * d) + s2[p:]) for p in self.gap_positions)

    def __score(self, s1, s2) -> float:
        x = 0
        for i in range(self.v_offset, len(s1) - self.j_offset):
            c1 = s1[i]
            c2 = s2[i]
            if c1 == '-' or c2 == '-':
                x = x + self.gap_penalty
            else:
                x = x + self.mat[c1, c2]
        return self._factor * x

    def alns(self, s1, s2) -> tuple[tuple[str, str, float]]:
        return tuple((sp1, sp2, self.__score(sp1, sp2)) for (sp1, sp2) in self.pad(s1, s2))

    def score(self, s1, s2) -> float:
        return max(self.__score(sp1, sp2) for (sp1, sp2) in self.pad(s1, s2))

    def score_norm(self, s1, s2) -> float:
        score1 = sum(self.mat[c, c] for c in s1)
        score2 = sum(self.mat[c, c] for c in s2)
        return self.score(s1, s2) - max(score1, score2)


class _Scoring_Wrapper:
    def __init__(self, scoring: Scoring):
        self.scoring = scoring

    def __call__(self, gs1: tuple[str, str], gs2: tuple[str, str]):
        return ((gs1[0], gs2[0]), self.scoring.score(gs1[1], gs2[1]))


class GermlineAligner:
    def __init__(self, dist: dict[tuple[str, str], float]):
        self.dist = dist
        self.dist.update(dict(((g2, g1), score)
                         for ((g1, g2), score) in dist.items()))

    def score(self, g1: str | Segment, g2: str | Segment) -> float:
        if isinstance(g1, Segment):
            g1 = g1.id
        if isinstance(g2, Segment):
            g2 = g2.id
        return self.dist[(g1, g2)]

    def score_norm(self, g1: str | Segment, g2: str | Segment) -> float:
        return self.score(g1, g2) - max(self.score(g1, g1), self.score(g2, g2))

    @classmethod
    def from_seqs(cls,
                  seqs: dict[str, str] | t.Iterable[tuple[str, str]] | list[Segment],
                  scoring: Scoring = BioAlignerWrapper(),
                  nproc=1, chunk_sz=4096):
        scoring_wrapper = _Scoring_Wrapper(scoring)
        if type(seqs) is dict:
            seqs = seqs.items()
        elif isinstance(seqs, list) and isinstance(seqs[0], Segment):
            seqs = dict({s.id, s.seqaa} for s in seqs)
        gen = ((gs1, gs2) for gs1 in seqs for gs2 in seqs if
               gs1[0] >= gs2[0])
        if nproc == 1:
            dist = starmap(scoring_wrapper, gen)
        else:
            with Pool(nproc) as pool:
                dist = pool.starmap(scoring_wrapper, gen, chunk_sz)
        return cls(dict(dist))


class ClonotypeScore:
    __scores__ = ['v_score', 'j_score', 'cdr3_score']

    def __init__(self, v_score: float, j_score: float, cdr3_score: float):
        self.v_score = v_score
        self.j_score = j_score
        self.cdr3_score = cdr3_score


class ClonotypeAligner:
    def __init__(self,
                 v_aligner: GermlineAligner,
                 j_aligner: GermlineAligner,
                 cdr3_aligner: CDRAligner = CDRAligner()):
        self.v_aligner = v_aligner
        self.j_aligner = j_aligner
        self.cdr3_aligner = cdr3_aligner

    @classmethod
    def from_library(cls,
                  lib: SegmentLibrary = SegmentLibrary.load_default(),
                  gene: str = None,
                  cdr3_aligner: CDRAligner = CDRAligner()):
        v_aligner = GermlineAligner.from_seqs(lib.get_seqaas(gene=gene, stype='V'))
        j_aligner = GermlineAligner.from_seqs(lib.get_seqaas(gene=gene, stype='J'))
        return cls(v_aligner, j_aligner, cdr3_aligner)

    def score(self, cln1: ClonotypeAA, cln2: ClonotypeAA) -> ClonotypeScore:
        return ClonotypeScore(v_score=self.v_aligner.score(cln1.v, cln2.v),
                              j_score=self.j_aligner.score(cln1.j, cln2.j),
                              cdr3_score=self.cdr3_aligner.score(cln1.cdr3aa, cln2.cdr3aa))

    def score_norm(self, cln1: ClonotypeAA, cln2: ClonotypeAA) -> ClonotypeScore:
        return ClonotypeScore(v_score=self.v_aligner.score_norm(cln1.v, cln2.v),
                              j_score=self.j_aligner.score_norm(
                                  cln1.j, cln2.j),
                              cdr3_score=self.cdr3_aligner.score_norm(cln1.cdr3aa, cln2.cdr3aa))
