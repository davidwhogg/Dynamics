ALLFILES = $(patsubst %.tex,%.pdf,$(wildcard *.tex))

all: $(ALLFILES)

%.pdf: %.tex
	pdflatex $<
	bash -c " ( grep Rerun $*.log && pdflatex $< ) || echo noRerun "
	bash -c " ( grep Rerun $*.log && pdflatex $< ) || echo noRerun "
