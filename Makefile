# Makefile pour la proposition de stage M2
# Compilation LaTeX avec pdflatex

# Nom des fichiers (sans extension)
MAIN = proposition-stage
LONG = proposition-stage-long

# Compilateur
LATEX = pdflatex
BIBTEX = bibtex

# Options de compilation
LATEXFLAGS = -interaction=nonstopmode -halt-on-error

# Fichiers générés à nettoyer
CLEANEXT = aux log out toc lof lot bbl blg nav snm vrb fls fdb_latexmk synctex.gz

.PHONY: all long clean distclean view install

# Installation des dépendances LaTeX (Ubuntu/Debian)
install:
	@echo "Installation des packages LaTeX nécessaires..."
	sudo apt-get update
	sudo apt-get install -y texlive-latex-base texlive-latex-recommended \
		texlive-latex-extra texlive-lang-french texlive-fonts-recommended \
		texlive-bibtex-extra latexmk
	@echo "Installation terminée."

# Cible par défaut : compilation complète
all: $(MAIN).pdf

# Compilation du PDF (version courte)
$(MAIN).pdf: $(MAIN).tex
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex

# Compilation de la version longue
long: $(LONG).pdf

$(LONG).pdf: $(LONG).tex
	$(LATEX) $(LATEXFLAGS) $(LONG).tex
	$(LATEX) $(LATEXFLAGS) $(LONG).tex

# Compilation avec bibliographie (si nécessaire)
bib: $(MAIN).tex
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	$(BIBTEX) $(MAIN)
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex

# Nettoyage des fichiers auxiliaires
clean:
	rm -f $(foreach ext,$(CLEANEXT),$(MAIN).$(ext))
	rm -f $(foreach ext,$(CLEANEXT),$(LONG).$(ext))

# Nettoyage complet (y compris les PDF)
distclean: clean
	rm -f $(MAIN).pdf $(LONG).pdf

# Ouvrir le PDF (Linux avec xdg-open, adapter si nécessaire)
view: $(MAIN).pdf
	xdg-open $(MAIN).pdf &

# Compilation continue (nécessite latexmk)
watch:
	latexmk -pdf -pvc $(MAIN).tex

# Aide
help:
	@echo "Cibles disponibles :"
	@echo "  all       - Compile la version courte (2 pages)"
	@echo "  long      - Compile la version longue (7 pages)"
	@echo "  bib       - Compile avec bibliographie BibTeX"
	@echo "  clean     - Supprime les fichiers auxiliaires"
	@echo "  distclean - Supprime tous les fichiers générés"
	@echo "  view      - Ouvre le PDF"
	@echo "  watch     - Compilation continue (nécessite latexmk)"
	@echo "  install   - Installe les dépendances LaTeX (sudo)"
	@echo "  help      - Affiche cette aide"
