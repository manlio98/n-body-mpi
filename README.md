# N-Body Parallel Programming


## Introduzione

Il problema N-Body consiste trovare le posizioni e le velocità di un insieme di particelle interagenti nel tempo. Ad esempio, un astrofisico potrebbe voler conoscere le posizioni e le velocità di un gruppo di stelle. Al contrario, un chimico potrebbe voler conoscere le posizioni e le velocità di un insieme di molecole o atomi.\newline Un programma che risolve il problema N-Body simula il comportamento delle particelle. L'input del problema è la massa, la posizione e la velocità di ciascuna particella all'inizio della simulazione e l'output è tipicamente la posizione e la velocità di ciascuna particella in sequenza di tempi specificati dall'utente, o semplicemente la posizione e la velocità di ciascuna particella al termine di un tempo specificato dall'utente.

## Breve descrizione della soluzione

Per risolvere il problema si è considerata la soluzione quadratica nel numero di particelle, però anche l'algoritmo Barnes-Hut può essere considerato, ma dovrebbe essere più difficile da sviluppare. Per la parte matematica, cioè nel calcolo della forza dei corpi, si è presa in considerazione la soluzione di Harrism (https://github.com/harrism/mini-nbody/blob/master/nbody.c). Il programma è in grado di simulare il processo per un determinato numero di cicli stabiliti dall'utente. Il processo MASTER inizializza un array di body in modo pseudocasuale e lo invia ai processori P-1. Nella nostra soluzione si è deciso che il processo MASTER contribuisce alla computazione, ma si potrebbe anche scegliere che il processo MASTER non partecipi al calcolo. Ogni SLAVE simula la forza dei corpi (bodyForce), solo per i suoi corpi, e invia i risultati dei suoi corpi a tutti gli altri processori, necessari per la fase successiva della simulazione. Al termine della simulazione ogni processo SLAVE invia il suo gruppo di body al processo MASTER il quale stamperà i risultati della simulazione.

## Dettagli dell’implementazione

In questa sezione si vedrà l'implementazione nei dettagli del problema n-body. Si inizia parlando di come si è rappresentato un singolo body, poi si parla di come si è suddiviso il problema tra i processori, e infine si parla della computazione e comunicazione di ogni processore.

### Definizione della struttura Body
Ogni body `e stato rappresentato con una struct, in questo modo:
``` c
typedef struct { 
  float x, y, z, vx, vy, vz; 
} Body;
```
La struttura è rappresentata da 6 valori float, i primi 3 valori (x,y e z) rappresentano la posizione, mentre i restanti (vx,vy e vz) rappresentano la velocità.

### Inizializzazione
Prima di inizializzare l'array di body, bisogna creare un nuovo tipo di dato in MPI in modo tale che il body possa essere inviato e ricevuto da un processore. Lo snippet di codice che effettua questo procedimento è il seguente:
``` c
MPI_Datatype body_type;
MPI_Type_contiguous(6,MPI_FLOAT,&body_type);
MPI_Type_commit(&body_type);
```
Si è definito una nuova variabile bodytype di tipo MPI_Datatype, poi viene chiamata la routine MPI_Type_contiguos che replica 6 volte il tipo di dato float in locazioni di memoria contigue, prima di essere usato nelle comunicazioni il tipo di dato nuovo deve essere commitato tramite la routine MPI_Type_commit.
Dopo aver creato il nuovo tipo di dato in MPI, si passa alla fase d'inizializzazione dove il processo MASTER in maniera pseudocasuale inizializza i body tramite la funzione randomizedBodies:
``` c
 if (world_rank == 0) {
    randomizeBodies(buf, 6*nBodies);
  }
```
La funzione randomizeBodies è definita in questo modo:
``` c
void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0 * (rand() / (float)RAND_MAX) - 1.0f;
  }
  
}
```
Dove il parametro n è indica il numero di body fornito in input al programma, mentre *datà è un puntatore di tipo float che punta all'array di body di dimensione n, che viene definito in questo modo:
``` c
int nBodies = atoi(argv[1]);
int bytes = nBodies*sizeof(Body);
float *buf = (float*)malloc(bytes);
Body *p = (Body*)buf;
```
### Suddivisione task
Una volta effettuata la fase d'inizializzazione, bisogna fare in modo che ogni processore riceva un carico di body  di dimensione minore rispetto all'input e dove simula la forza dei corpi  tramite la funzione bodyForce e BodyForceEsclude che verranno presentate in seguito. \newline Per dividere i body tra i processori, si è utilizzato la routine:
``` c
int MPI_Scatterv(const void* buffer_send,
const int counts_send[],
const int displacements[], 
MPI_Datatype datatype_send,
void* buffer_recv,
int count_recv,
MPI_Datatype datatype_recv,
int root,
MPI_Comm communicator); 
```
Dove:
- **buffer_send:** Il buffer contenente i dati da inviare agli altri processori. 
- **counts_send:** Un array che contiene il numero di elementi da inviare a ciascun processore.
- **displacements:** Un array contenente lo spostamento da applicare al messaggio inviato a ciascun processore..
- **datatype_send:** il tipo di dato del buffer di invio.
- **buffer_recv:** il buffer in cui archiviare i dati inviati.
- **count_recv:** Il numero di elementi nel buffer di ricezione.
- **datatype_recv:** il tipo di dato del buffer di ricezione.
- **root:** il rank del processore che invierà i dati ai processori.
- **communicator:** il communicatore in cui avviene la scatter.

Tramite il seguente snippet di codice si sono calcolati i vari parametri della routine MPI_Scatterv:
``` c
int rest = nBodies % (world_size); 
int portion = nBodies / (world_size); 
int send_counts[world_size];
int offset[world_size];
int sum = 0;
for (int i = 0; i < world_size;i++) {
    send_counts[i] = portion;
    if (rest > 0) {
      send_counts[i]++;
      rest--;
    }
    offset[i] = sum;
    sum+= send_counts[i];
  }
```

Questo snippet di codice sfrutta che il resto della divisione è sempre minore del divisore. Nel caso in cui il resto è pari a zero, ogni processore riceve la stessa porzione di body e la porzione viene definita nella variabile **portion**. Nel caso in cui il resto non è zero, significa che l'input non è divisibile per il numero di processori, quindi lo snippet di codice aggiunge un body in più al processore i-esimo,  finché non si raggiunge con la variabile **rest** il valore zero, questo ci permette di computare tutti i body e non escludere dalla computazione un numero di body pari al resto.Supponiamo di effettuare un esempio avendo in input un numero di body uguale a 10 e 3 processori, la suddivisione dei task sarebbe 4 body al primo processore, 3 body al secondo processore e 3 body al terzo processore

### Computazione e Comunicazione tra i processi

Terminata la fase di inzializzazione, inzia la fase in cui ogni processo possiede i suoi body  e calcola le forze dei corpi usando bodyForce e bodyForceEsclude. Innanzitutto, si è deciso di utilizzare un tipo di comunicazione non bloccante, perché in questo modo non bisogna attendere il completamento della comunicazione, ma si può compiere altre operazioni e nel nostro caso è la funzione bodyForce che effettua il  calcolo delle forze sui body. La comunicazione non bloccante utilizzata è la routine di MPI denominata **MPI\_Iallgatherv**  che permette di inviare i body di appartenenza a tutti gli altri processori in modo tale da completare la computazione. Dato che si tratta di una comunicazione non bloccante, come ho detto prima, in attesa che la comunicazione venga completata si è sfruttata questo tempo nel calcolare le forze sui body di appartenenza che il processore possiede. Lo snippet di codice che effettua questa fase è il seguente:
``` c
MPI_Iallgatherv(&p[offset[world_rank]],
send_counts[world_rank],
body_type,
p,
send_counts,
offset,body_type,
MPI_COMM_WORLD,
&request);
bodyForce(&p[offset[world_rank]],dt,send_counts[world_rank]);
```
Lo snippet di codice  utilizza **MPI_Iallgatherv** che riceve come argomenti l'indirizzo dei body da inviare, il numero di body posseduti dal processore, il tipo di dato che si invia, il buffer in cui archiviare i body raccolti, un array contenente il numero di body da ricevere da ogni processore, un array contenente l' indice di partenza(displacements) dei body ricevuti, il tipo di dato che si riceve, il comunicatore, e infine la variabile in cui archiviare il gestore per determinare se la richiesta è stata completata o meno. Dopodiché, si chiama la funzione bodyForce per iniziare a calcolare le forze sui body di appartenenza, infatti si passa in input alla funzione i body, un float espresso con la costante DT nel programma che rappresenta il time step e infine la dimensione totale dei body posseduti dal processore. In seguito, si vede nel dettaglio la funzione bodyForce:
``` c
void bodyForce(Body *p, float dt, int n_body) {
  for (int i = 0; i < n_body; i++) { 
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;
    for (int j = 0; j < n_body; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = 1.0f / sqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;
  
      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }
    p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
  }
}
```
Dopo che ogni processore calcola le forze sui body, il programma attende che la comunicazione (MPI\_Iallgatherv) tra i processori termini e ciò possibile fare tramite la routine **MPI\_Wait}:**
``` c
MPI_Wait(&request,&status);
```
La funzione **MPI_Wait** attende quindi il completamento di un'operazione non bloccante. Dopo l'attesa siamo sicuri che il processore ha ricevuto da tutti gli altri processori i body in modo tale da permettere al processore di completare la computazione, una cosa importante in questa fase che il processore deve escludere dal calcolo i body di sua appartenenza, calcolati in precedenza, ma loro devono interagire con i body ricevuti da altri processori. Lo snippet di codice che permette di fare ciò:
``` c
bodyForceEsclude(p,dt,nBodies,offset[world_rank],send_counts[world_rank]);
```
La funzione bodyForceEsclude prende come input i body, la costante float dt, il numero di body totale, l'indice di partenza dei body di appartenenza del processore, il numero di body posseduti dal processore. 
In seguito, si può notare nel dettaglio la funzione bodyForceEsclude:
``` c
void bodyForceEsclude(Body *p, float dt, int n,int offset_start,int portion) {

  for (int i = offset_start; i < offset_start + portion; i++) { 
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;
      for (int j = 0; j < offset_start; j++) {
    
          float dx = p[j].x - p[i].x;
          float dy = p[j].y - p[i].y;
          float dz = p[j].z - p[i].z;
          float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
          float invDist = 1.0f / sqrtf(distSqr);
          float invDist3 = invDist * invDist * invDist;


          Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
        
      }
        p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
  }

  for (int i = offset_start; i < offset_start + portion; i++) { 
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;
      for (int j = offset_start + portion; j < n; j++) {
    
          float dx = p[j].x - p[i].x;
          float dy = p[j].y - p[i].y;
          float dz = p[j].z - p[i].z;
          float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
          float invDist = 1.0f / sqrtf(distSqr);
          float invDist3 = invDist * invDist * invDist;

          Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
        
      }
        p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
  }
}
```
La funzione BodyForceEsclude è molto simile alla funzione bodyForce precedente, ma l'unica differenza permette di escludere dalla computazione i body calcolati in precedenza, e si può notare dai for, perché il primo parte da 0 fino al primo body di apparteneneza del processore, mentre il secondo for parte dall'ultimo body di appartenenza del processore e termina fino all'ultimo body. Dopo aver fatto questo bisogna fare un ultimo passo, cioè quello di aggiornare le posizioni dei body di appartenenza in quanto hanno interagito con gli altri body. Lo snippet di codice è il seguente:
``` c
for (int i = 0 ; i < send_counts[world_rank]; i++) { // update position
      (&p[offset[world_rank]])[i].x += (&p[offset[world_rank]])[i].vx*dt;
      (&p[offset[world_rank]])[i].y += (&p[offset[world_rank]])[i].vy*dt;
      (&p[offset[world_rank]])[i].z += (&p[offset[world_rank]])[i].vz*dt;

}
```
Questo è ultimo passo per la computazione, una volta che sono terminate le iterazioni ogni processore invia i body di appartenenza al processore MASTER tramite la routine MPI **MPI_Gatherv** come segue:
``` c
MPI_Gatherv(&p[offset[world_rank]],
send_counts[world_rank],
body_type,
p,
send_counts,
offset,
body_type,
0,
MPI_COMM_WORLD);
```
Alla funzione riceve come input:
- **p[\&offset[world_rank]]:** L'indirizzo iniziale dei body di appartenenza. 
- **send_counts[world_rank]:** il numero di body appartenenti al processore.
- **body_type:** il tipo di dato body\_type (struct) inviato.
- **p**: il buffer dove viene raccolto tutti i body.
- **send_counts: l'array che contiene il numero di body inviati da ciascun processore.
- **offset:** l'array che contiene l'indice di partenza di ogni processore.
- **body_type:** il tipo di dato body\_type (struct) ricevuto.
- **0**: il processore MASTER raccoglie i body.
- **MPI_COMM_WORLD:** il communicatore.
A questo punto il processore MASTER salva in un file .txt il tempo di esecuzione e i risultati della computazione.


## Istruzioni per l'esecuzione
Per poter lanciare il programma, bisogna effettuare prima la fase di compilazione utilizzando **mpicc** come segue:
``` bash
mpicc -g n-body.c -o n-body -lm
```
Una volta eseguito il comando precedente, è possibile eseguire il programma utilizzando questa volta **mpirun** come segue:
``` bash
mpirun --allow-run-as-root -np P n-body B I
```
Dove:
- **P**: il numero di processori da utilizzare.
- **B**: il numero di body in input.
- **I:** il numero di iterazioni in input.
Esempio di utilizzo è il seguente:
``` bash
mpirun --allow-run-as-root -np 10  n-body 100  5 
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
