# maskinlaering
Det er 5 forskjellige kodefiler som er brukt for å løse oppgaven. Det er en decision_tree_for_forest.py som er filen med det oppdaterte versjonen av Decision Tree for Random Forest (altså med max_features). 
Den andre koden er random_forest.py. Dette er koden for en Random Forest som er implementert med decision_tree_for_forest. 
Den tredje er Grid_Search_With_Kfold.py. Dette er funksjonen vi bruker for å tune hyperparameterene til modelen. Grunnen for at vi lagde denne isteden for å bruke scikit sin innebygde var for å nærmere forstå selve utvelgelsen av hyperparameterene. Det kan være at den er mindre effektiv, men resultatene er fortsatt bra. 
Den fjerde er en plot_function som er en enkel plotte funksjon som vi brukte i starten for å visualisere dataene.
INGEN av de 4 filene over gir noe print eller lignende for å gi svar til oppgaven. Disse er alle classer og funksjoner som blir brukt i run.ipynb for å besvare oppgaven. 
run.ipynb er da run filen. Denne kan man kjøre gjennom for å få resultatene.  

Siden 'DecisionTree' er en modifisert for Random Forest vil den til dels unvike fra en normal ID3 algoritme. For eksempel at den velger et delset av features som vi har lagt til for random forest for å unvike 'overfitting'.