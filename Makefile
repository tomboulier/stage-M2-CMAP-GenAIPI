# Makefile pour la proposition de stage M2
# Compilation LaTeX avec pdflatex

# Nom du fichier principal (sans extension)
MAIN = proposition-stage

# Compilateur
LATEX = pdflatex
BIBTEX = bibtex

# Options de compilation
LATEXFLAGS = -interaction=nonstopmode -halt-on-error

# Fichiers générés à nettoyer
CLEANEXT = aux log out toc lof lot bbl blg nav snm vrb fls fdb_latexmk synctex.gz

.PHONY: help pdf clean distclean view install check-deps

# Cible par défaut : affiche l'aide
.DEFAULT_GOAL := help

# Aide
help:
	@echo "Proposition de stage M2 - CMAP GenAIPI"
	@echo ""
	@echo "Cibles disponibles :"
	@echo "  make pdf      - Compile le document PDF"
	@echo "  make view     - Compile et ouvre le PDF"
	@echo "  make clean    - Supprime les fichiers auxiliaires"
	@echo "  make distclean- Supprime tous les fichiers générés"
	@echo "  make install  - Installe les dépendances LaTeX (sudo)"
	@echo "  make help     - Affiche cette aide"

# Vérification des dépendances
check-deps:
	@which pdflatex > /dev/null || (echo "Erreur: pdflatex non trouvé. Lancez 'make install'" && exit 1)

# Installation des dépendances LaTeX (Ubuntu/Debian)
install:
	@echo "Installation des packages LaTeX nécessaires..."
	sudo apt-get update
	sudo apt-get install -y texlive-latex-base texlive-latex-recommended \
		texlive-latex-extra texlive-lang-french texlive-fonts-recommended \
		texlive-bibtex-extra latexmk
	@echo "Installation terminée."

# Compilation du PDF
pdf: check-deps $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex

# Nettoyage des fichiers auxiliaires
clean:
	rm -f $(foreach ext,$(CLEANEXT),$(MAIN).$(ext))

# Nettoyage complet (y compris le PDF)
distclean: clean
	rm -f $(MAIN).pdf

# Ouvrir le PDF (Linux avec xdg-open)
view: pdf
	xdg-open $(MAIN).pdf &
