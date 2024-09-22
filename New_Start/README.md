# maskinlaering
Det er 4 forskjellige kodefiler som er brukt for å løse oppgaven. Det er en decision_tree_for_forest.py som er filen med det oppdaterte versjonen av Decision Tree for Random Forest (altså med max_features). 
Den andre koden er random_forest.py. Dette er koden for en Random Forest som er implementert med decision_tree_for_forest. 
Den tredje er function_implimentations.py. Dette er en python fil med funksjoner vi har brukt for å løse oppgaven i run filen som verken hører hjemme i decision tree eller random forest filen. Disse funksjonene er da grid_search, plot og test.
run.ipynb er da run filen. Denne kan man kjøre gjennom for å få resultatene.  

Siden 'DecisionTree' er en modifisert for Random Forest vil den til dels unvike fra en normal ID3 algoritme. For eksempel at den velger et delset av features som vi har lagt til for random forest for å unvike 'overfitting'.