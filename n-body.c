#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


#define SOFTENING 1e-9f

typedef struct { 
  float x, y, z, vx, vy, vz; 
} Body;

void randomizeBodies(float *data, int n) {
  srand(1);
  for (int i = 0; i < n; i++) {
    data[i] = 2.0 * (rand() / (float)RAND_MAX) - 1.0f;
  }
  
}

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


void bodyForce(Body *p, float dt, int n_body) {
  for (int i = 0; i < n_body; i++) { 
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;
    for (int j = 0; j < n_body; j++) { //j deve partire sempre da zero
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

void save_bodies(Body *p,int n_bodies) {
  FILE *out_parallel = fopen("body_parallel.txt","w");
  fprintf(out_parallel,"Body   :     x              y               z           |           vx              vy              vz   \n");
  for (int i = 0; i < n_bodies; i++) {
    fprintf(out_parallel,"Body %d : %lf\t%f\t%lf\t|\t%lf\t%lf\t%lf\n",i+1,p[i].x,p[i].y,p[i].z,p[i].vx,p[i].vy,p[i].vz);
  }
}





int main(int argc, char** argv) {

  MPI_Init(&argc,&argv);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Status status;
  MPI_Request request;
  /* CREO IL TIPO IN MPI (STRUCT BODY)  */
  MPI_Datatype body_type; //creare il tipo
  MPI_Type_contiguous(6,MPI_FLOAT,&body_type);
  MPI_Type_commit(&body_type);

  /* input del problema*/
  int nBodies = atoi(argv[1]);
  int n_cycle = atoi(argv[2]);;
  const float dt = 0.01f;
  int bytes = nBodies*sizeof(Body);
  float *buf = (float*)malloc(bytes);
  Body *p = (Body*)buf;

  if (world_rank == 0) {
    randomizeBodies(buf, 6*nBodies); //fase di inizializzazione. Ogni body viene inzializzato in maniera pseudocasuale
  }

  
  /*calcolo send_count e displacement */
  int rest = nBodies % (world_size); //calcolo il resto
  int portion = nBodies / (world_size); //calcola la porzione da dare agni processore
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


  
  MPI_Barrier(MPI_COMM_WORLD);
  double start= MPI_Wtime();
  //dividi l'array p (body) come descritto nella send_counts e offset
  MPI_Scatterv(p,send_counts,offset,body_type,&p[offset[world_rank]],send_counts[world_rank],body_type,0,MPI_COMM_WORLD);
  for (int iter = 1; iter <= n_cycle; iter++) {
    MPI_Iallgatherv(&p[offset[world_rank]],send_counts[world_rank],body_type,p,send_counts,offset,body_type,MPI_COMM_WORLD,&request);
    /*Computa bodyForce sui body di appartenenza*/
    bodyForce(&p[offset[world_rank]],dt,send_counts[world_rank]);
   
    //attesa che mpi_allgather termini
    MPI_Wait(&request,&status);
    //bodyforce escludendo la parte già calcolata
    bodyForceEsclude(p,dt,nBodies,offset[world_rank],send_counts[world_rank]);
    /*update position*/
    for (int i = 0 ; i < send_counts[world_rank]; i++) { // update position
      (&p[offset[world_rank]])[i].x += (&p[offset[world_rank]])[i].vx*dt;
      (&p[offset[world_rank]])[i].y += (&p[offset[world_rank]])[i].vy*dt;
      (&p[offset[world_rank]])[i].z += (&p[offset[world_rank]])[i].vz*dt;

    } 
  }
  /*ogni processo invia i body di appartenenza al processo master*/
  MPI_Gatherv(&p[offset[world_rank]],send_counts[world_rank],body_type,p,send_counts,offset,body_type,0,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  double end = MPI_Wtime();
  if (world_rank == 0) {
    save_bodies(p,nBodies); //salva i risulati in un file txt 
    double time_execution = end - start;
    FILE *file = fopen("./body_execution_time.txt","a");
    fprintf(file,"Con %d processori, %d body e %d cicli il tempo di esecuzione è %fs\n",world_size,nBodies,n_cycle,time_execution);

  }
  free(buf);
  MPI_Type_free(&body_type);

  MPI_Finalize();
}