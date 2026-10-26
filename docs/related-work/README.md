# Related work: facility layout classification

Materials prepared for the ME5001A report's **Related Work** section.

## Read the documents

| File | Contents |
|---|---|
| [English LaTeX](framework.tex) | Classification diagram, explanatory prose, comparison of five sources, and numbered references |
| [中文 PDF](zh/framework.pdf) | 三页中文版：黑白分类图、配套说明、文献比较表和参考文献 |
| [中文 LaTeX](zh/framework.tex) | 中文版的可编辑源稿 |

The Chinese PDF uses vector outlines for text because some viewers failed to display the embedded Chinese fonts in the original export. It remains sharp when enlarged and printed, but text selection and search are unavailable in this viewing copy. Use the LaTeX source for editing or copying the prose.

## What the diagram represents

This is a selective synthesis of the sources below, rather than a reproduction of one published figure or an exhaustive taxonomy. It separates:

1. Problem characteristics: layout units, department geometry, temporal setting and planning context.
2. Objective criteria.
3. Layout representations.
4. Mathematical formulations and relaxations.
5. Solution approaches, with metaheuristics under the broader heuristic family.
6. Spatial structure and multi-floor decisions.
7. Layout constraints.
8. Data assumptions and performance assessment.

These dimensions can coexist in one study. Layout encoding, mathematical formulation, search and performance evaluation serve different roles. The diagram does not establish an algorithm ranking or claim that every method covers every FLP variant.

## Literature index

| Source | Reference | Role in this synthesis |
|---|---|---|
| Drira, Pierreval and Hajri-Gabouj (2007) | “Facility layout problems: A survey,” *Annual Reviews in Control*, 31(2), 255–267. [DOI](https://doi.org/10.1016/j.arcontrol.2007.04.001) | General problem and method classification |
| Pérez-Gosende, Mula and Díaz-Madroñero (2021) | “Facility layout planning. An extended literature review,” *International Journal of Production Research*, 59(12), 3777–3816. [DOI](https://doi.org/10.1080/00207543.2021.1897176) | Planning conditions, layout generation and assessment |
| Anjos and Vieira (2017) | “Mathematical optimization approaches for facility layout problems: The state-of-the-art and future research directions,” *European Journal of Operational Research*, 261(1), 1–16. [DOI](https://doi.org/10.1016/j.ejor.2017.01.049) | Formulations, relaxations and mathematical optimisation |
| Ahmadi, Pishvaee and Akbari Jokar (2017) | “A survey on multi-floor facility layout problems,” *Computers & Industrial Engineering*, 107, 158–170. [DOI](https://doi.org/10.1016/j.cie.2017.03.015) | Floor assignment and vertical transport |
| Liu (2004) | “A sequence-pair and mixed integer programming based methodology for the facility layout problem,” Ph.D. dissertation, Virginia Tech. [Institutional full text](https://vtechworks.lib.vt.edu/server/api/core/bitstreams/34c76035-26bc-4ee1-a580-86711cd7b4c9/content) | Sequence-pair search combined with mathematical programming |

Both LaTeX documents include IEEE-style numbered references. When incorporating the figure and prose into the report, merge the citation keys into the report's bibliography and let the report determine numbering. Attribute the diagram as a synthesis of the cited literature.

## Relationship to the benchmark repository

The literature overview covers a wider range of problems, including multi-floor layouts, than the repository's current [experimental protocol](../protocol.md), which specifies static, single-floor, rectangular-department UA-FLP. The overview does not change that protocol or the [frozen instance manifest](../../instances/MANIFEST.md). It does not report new algorithm reproductions or benchmark results.

## Build from source

A TeX installation with TikZ is required; the Chinese source also needs `ctex` and the Fandol fonts. Run each command twice to resolve references.

From this directory:

```sh
pdflatex -interaction=nonstopmode -halt-on-error framework.tex
pdflatex -interaction=nonstopmode -halt-on-error framework.tex
```

From the `zh/` subdirectory:

```sh
xelatex -interaction=nonstopmode -halt-on-error framework.tex
xelatex -interaction=nonstopmode -halt-on-error framework.tex
```

Those commands produce ordinary text PDFs. To create the font-independent Chinese viewing copy with Ghostscript, keep distinct input and output paths:

```sh
gs -q -dSAFER -dBATCH -dNOPAUSE -sDEVICE=pdfwrite \
  -dCompatibilityLevel=1.4 -dNoOutputFonts -dAutoRotatePages=/None \
  -sOutputFile=framework-outlined.pdf framework.pdf
```

Retain the outlined output as the distributed `zh/framework.pdf` after checking all pages. Generated auxiliary files and temporary build products are ignored.
