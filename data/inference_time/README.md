# RAM usage 

nei file ..._first_question.csv ci sono i risultati della RAM usage. la colonna pricipale è proc_rss_mb_peak_during_inference che in MB indica l'uso della RAM.
I valori sono secondo me ragionevoli pensando alla grandezza dei Modelli più reranker embedder model e il vector store più il normale consumo di altre app aperte (es. browser).

Il modello qwen 3b è quantizzato in int8 qanche se nel nome del file non c'è scritto.
Il modello phi non si prova mai la versione non quatizzata in questi test.