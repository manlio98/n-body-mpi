# n N-Body Parallel Programming


## Introduzione

Il problema N-Body consiste trovare le posizioni e le velocit`a di un insieme di particelle interagenti
nel tempo. Ad esempio, un astrofisico potrebbe voler conoscere le posizioni e le velocit`a di un gruppo
di stelle. Al contrario, un chimico potrebbe voler conoscere le posizioni e le velocit`a di un insieme di
molecole o atomi.
Un programma che risolve il problema N-Body simula il comportamento delle particelle. L’input del
problema `e la massa, la posizione e la velocit`a di ciascuna particella all’inizio della simulazione e
l’output `e tipicamente la posizione e la velocit`a di ciascuna particella in sequenza di tempi specificati
dall’utente, o semplicemente la posizione e la velocit`a di ciascuna particella al termine di un tempo
specificato dall’utente

## Breve descrizione della soluzione

Per risolvere il problema si `e considerata la soluzione quadratica nel numero di particelle, per`o anche
l’algoritmo Barnes-Hut pu`o essere considerato, ma dovrebbe essere pi`u difficile da sviluppare. Per la
parte matematica, cio`e nel calcolo della forza dei corpi, si `e presa in considerazione la soluzione di
Harrism.
Il programma `e in grado di simulare il processo per un determinato numero di cicli stabiliti dall’utente.
Il processo MASTER inizializza un array di body in modo pseudocasuale e lo invia ai processori P1. Nella nostra soluzione si `e deciso che il processo MASTER contribuisce alla computazione, ma si
potrebbe anche scegliere che il processo MASTER non partecipi al calcolo. Ogni SLAVE simula la forza
dei corpi (bodyForce), solo per i suoi corpi, e invia i risultati dei suoi corpi a tutti gli altri processori,
necessari per la fase successiva della simulazione. Al termine della simulazione ogni processo SLAVE
invia il suo gruppo di body al processo MASTER il quale stamper`a i risultati della simulazione.

## Dettagli dell’implementazione

In questa sezione si vedr`a l’implementazione nei dettagli del problema n-body. Si inizia parlando di
come si `e rappresentato un singolo body, poi si parla di come si `e suddiviso il problema tra i processori,
e infine si parla della computazione e comunicazione di ogni processore.

### Definizione della struttura Body
Ogni body `e stato rappresentato con una struct, in questo modo:
'''c
typedef struct { 
  float x, y, z, vx, vy, vz; 
} Body;
'''


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
