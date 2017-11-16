.PHONY: all write latex wc hglog clean

BASENAME = main
CONTENT_FILES = main.tex

BASE_LATEX_COMMAND = lualatex -interaction=nonstopmode -halt-on-error -recorder -shell-escape

# If "pydflatex" is installed, use it to parse the logfile and print more useful output:
PYDFLATEX := $(shell command -v pydflatex 2> /dev/null)
ifdef PYDFLATEX
	PYDFLATEX_COMMAND = echo '====== BEGIN LATEX LOG ====='; pydflatex -l $(BASENAME).tex; echo '======= END LATEX LOG ======\n'
    LATEX_COMMAND = $(BASE_LATEX_COMMAND) $(BASENAME).tex > /dev/null; $(PYDFLATEX_COMMAND)
    LATEXMK_COMMAND = $(BASE_LATEX_COMMAND) %O %S > /dev/null; $(PYDFLATEX_COMMAND)
else
    LATEX_COMMAND = $(BASE_LATEX_COMMAND) $(BASENAME).tex
    LATEXMK_COMMAND = $(BASE_LATEX_COMMAND) %O %S
endif

# Tell TeX to just print lines as long as it wants:
export max_print_line=1048576

all:
	@# make with latexmk - runs latex and bibtex repreatedly as required
	@latexmk -silent -pdflatex="$(LATEXMK_COMMAND)" $(BASENAME).tex -pdf

write:
	@# latexmk in continuous view mode, running as necessary whenever files change
	@latexmk -silent -pvc -pdflatex="$(LATEXMK_COMMAND)" $(BASENAME).tex -pdf

latex:
	@# Just run latex by itself.
	@$(LATEX_COMMAND)

wc:
	@# Wordcount:
	@texcount -total $(CONTENT_FILES) | tee wc.txt

hglog:
	@# Write some info about the last commit to a file to be read into tex and printed as a footer in drafts:
	@hg log -l1 --template 'rev:     {rev} ({node|short})\nauthor:  {author}\ndate:    {date|date}\nsummary: {desc}' > hglog.txt

clean:
	@# Delete temporary files
	@rm -rf *.aux *.log *.out *.toc *.bbl *.blg *.brf *.fls *.fdb_latexmk hglog.txt _minted-$(BASENAME)/




