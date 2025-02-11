# Ottimizzazione dei Percorsi di Distribuzione Energetica

## Contesto del Problema

Una società di distribuzione energetica deve ottimizzare la manutenzione e il rifornimento di 20 stazioni di ricarica elettrica in una rete urbana interconnessa.

## Fasi del Progetto

### Fase 1: Sviluppo dell'Euristica

- Implementare una **heuristic function** personalizzata per l'algoritmo **A***
- Considerare criteri come:
    - Minimizzazione dei costi di attraversamento
    - Efficienza energetica
    - Criticità delle stazioni

### Fase 2: Simulazione dei Percorsi

Verranno simulati 3 diversi scenari di percorso:

1. **Scenario di Emergenza**
    - Punto di partenza: Stazione centrale
    - Obiettivo: Raggiungere la stazione più critica nel minor tempo possibile
    - Vincoli: Minimizzare il consumo energetico
2. **Scenario di Manutenzione Programmata**
    - Punto di partenza: Stazione nord
    - Obiettivo: Visitare 3 stazioni specifiche in un ordine predefinito
    - Vincoli: Ottimizzare il percorso complessivo
3. **Scenario di Bilanciamento Energetico**
    - Punto di partenza: Stazione sud
    - Obiettivo: Collegare 4 stazioni con basso livello di energia
    - Vincoli: Minimizzare la distanza totale percorsa

## Output Richiesti per Ogni Scenario

- Percorso ottimale trovato
- Costo totale del percorso
- Nodi attraversati in sequenza
- Analisi delle scelte dell'euristica
- Confronto con percorsi alternativi

## Criteri di Valutazione

- Correttezza implementazione **A***
- Efficacia della **heuristic function**
- Performance dei percorsi calcolati
- Capacità di adattamento a scenari diversi

## Requisiti Tecnici

- Implementare gestione dei pesi degli archi
- Dimostrare flessibilità dell'euristica
- Produrre una relazione dettagliata che spieghi le scelte implementative
- Documentare il processo di sviluppo dell'euristica
- Spiegare i criteri di ottimizzazione utilizzati
- Fornire visualizzazione grafica dei percorsi

## Esecuzione del main.py

```
python main.py --scenario emergency

python main.py --scenario maintenance

python main.py --scenario balancing

python main.py --help
```