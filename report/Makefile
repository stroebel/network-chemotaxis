# LaTeX Makefile
# Set MAIN to your .tex filename (without extension)
MAIN = main

PDF = $(MAIN).pdf
TEX = $(MAIN).tex
BIB = $(wildcard *.bib)

.PHONY: all clean

all: $(PDF)

$(PDF): $(TEX) $(BIB)
	pdflatex $(TEX)
	biber $(MAIN)
	pdflatex $(TEX)
	pdflatex $(TEX)

clean:
	rm -f $(MAIN).{aux,bbl,blg,log,out,pdf,toc,lof,lot}
