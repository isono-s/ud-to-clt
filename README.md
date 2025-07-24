# Calculate CLT/DLT costs from UD
This repository provides code for calculating processing difficulty metrics based on Category Locality Theory (CLT; [Isono, 2024](https://doi.org/10.1016/j.cognition.2024.105766)) and Dependency Locality Theory (DLT; [Gibson, 2000](https://tedlab.mit.edu/tedlab_website/researchpapers/Gibson_2000_DLT.pdf)) from [Universal Dependencies](https://universaldependencies.org/) annotation (CoNNL-U format).

I am providing a **UD**-to-CLT conversion though CLT is based on CCG rather than an conversion from an existing CCG parse because the variation in the implementation of CCG (especially the choice of unary rules) can have a significant effect on the CLT values. Note that the approach taken here is different from the original paper (Isono, 2024), which calculates CLT costs from Penn Treebank annotation, because UD-annotation is more readily available across languages. We are preparing a paper describing the details of the UD-based conversion.

## Requirements
- Python 3.7+
- Install dependencies using:

```bash
pip install -r requirements.txt
```

## Usage
```bash
python ud-to-clt.py /path/to/ud.conllu \
   --output /path/to/output.tsv \
   --headfinal \
   --savetrees
```
Add `--headfinal` if you are working on a head-final language, and add `--savetrees` will save CCG trees (both before and after incrementalization) in `./parses/`. Checking trees before using CLT values is highly recommended.

## License
MIT