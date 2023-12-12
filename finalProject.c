#include <stdlib.h>
#include <stdio.h> 
#include <math.h> 
#include <sys/time.h>
#include <omp.h>
// Authors: Jaime M. Orta, Christopher Torres, Mateo I. Muñiz
// Course: CIIC5019 - 001D
// Final Project

// Function to calculate time 
double get_walltime() {
struct timeval tp; gettimeofday(&tp, NULL);
return (double) (tp.tv_sec + tp.tv_usec*1e-6); }

// Displacement structure with x and y
typedef struct {
    float x;
    float y;
} Displacement;

// Particle structure, simulation assumes the International System of Units 
typedef struct {
    float x;    // X axis displacement
    float y;    // Y axis displacement
    float mass;    // Mass of the particle
    float time;    // Total time
    double velocity; // Velocity of the particle
    double kineticEnergy; // Kinetic energy of the particle
    double potentialEnergy; // Potential energy of the particle
} Particle;

// Shuffles unique displacements
void shufflePositions(Displacement *displacements, int count) {
    for (int i = count - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        Displacement temp = displacements[i];
        displacements[i] = displacements[j];
        displacements[j] = temp;
    }
}

 // Used to initialize the properties of the particles, it generates a random mass, time, displacement on x and displacement on y assuming all particles have a starting position of (0,0) and then move.
void initializeParticles(Particle *particles, int numParticles, double boxLength, int gridSize) {

    Displacement *displacments = (Displacement *)malloc(gridSize * gridSize * sizeof(Displacement));

    // Generate all possible displacements inside the box to ensure uniqueness since in a 2D simulation 2 particles can't exist in the same coordinates.
    for (int x = 0; x < gridSize; x++) {
        for (int y = 0; y < gridSize; y++) {
            displacments[x * gridSize + y].x = (float)x / gridSize * boxLength;
            displacments[x * gridSize + y].y = (float)y / gridSize * boxLength;
        }
    }

    // Shuffles displacements
    shufflePositions(displacments, gridSize * gridSize);

    // Assigns properties
    for (int i = 0; i < numParticles; i++) {
        particles[i].x = displacments[i].x;
        particles[i].y = displacments[i].y;
        particles[i].mass = 1 + (float) rand() / (RAND_MAX / 4.0); // Random float number between 1 and 4
        particles[i].time = 1 + (float) rand() / (RAND_MAX / 4.0); // Random float number between 1 and 4
        // printf("x: %f and y: %f\n", particles[i].x, particles[i].y); // Pueden comentarlo, solo es para verificar las coordenadas
        
    }

    free(displacments);
}

// Used to compare energies by adding up the energies of each implementation
double calculateTotalEnergy(Particle *particles, int numParticles, char energyType) {
    double totalEnergy = 0.0;
    for (int i = 0; i < numParticles; i++) {
        if (energyType == 'K') { // Kinetic energy
            totalEnergy += particles[i].kineticEnergy;
        } else if (energyType == 'P') { // Potential energy
            totalEnergy += particles[i].potentialEnergy;
        }
    }
    return totalEnergy;
}

// Function to compare the energies of the sequential and parallel implementations
void equalEnergies(double sequential, double stat, double dynamic, double guided){
    double min = 1e-6;
    if ( fabs(sequential - stat) > min || fabs(sequential - dynamic) > min || fabs(sequential - guided) > min){
        printf("Energies are not equal\n");
        return;
    }
    printf("Energies are equal\n");
    return;
    
}

// Calculates the velocity on x and y, and then the magnitude of the velocity vector
double velocity(float x, float y, float time){
    double velocityX = x / time;
    double velocityY = y / time;
    double velocityMagn = sqrt(pow(velocityX, 2.) + pow(velocityY, 2.));
    return velocityMagn;
}

// Calculates the kinetic energy, potential energy and velocity for each particle
void kinetic_potential_velocity_sequential(Particle *particles, int numParticles){
    double gravity = 9.81;
    for (int i = 0; i < numParticles; i++){
        particles[i].velocity = velocity(particles[i].x, particles[i].y, particles[i].time);
        particles[i].kineticEnergy = (particles[i].mass * pow(particles[i].velocity, 2.))/2.;
        particles[i].potentialEnergy = particles[i].mass * gravity * particles[i].y;
    }
}

// Calculates the kinetic energy, potential energy and velocity for each particle in parallel using guided schedule
void kinetic_potential_velocity_guided(Particle *particles, int numParticles, int chunk){
    double gravity = 9.81;
    #pragma omp parallel for schedule(guided, chunk) 
    for (int i = 0; i < numParticles; i++){
        particles[i].velocity = velocity(particles[i].x, particles[i].y, particles[i].time);
        particles[i].kineticEnergy = (particles[i].mass * pow(particles[i].velocity, 2.))/2.;
        particles[i].potentialEnergy = particles[i].mass * gravity * particles[i].y;
    }
}

// Calculates the kinetic energy, potential energy and velocity for each particle in parallel using static schedule
void kinetic_potential_velocity_static(Particle *particles, int numParticles){
    double gravity = 9.81;
    #pragma omp parallel for
    for (int i = 0; i < numParticles; i++){
        particles[i].velocity = velocity(particles[i].x, particles[i].y, particles[i].time);
        particles[i].kineticEnergy = (particles[i].mass * pow(particles[i].velocity, 2.))/2.;
        particles[i].potentialEnergy = particles[i].mass * gravity * particles[i].y;
    }
}

// Calculates the kinetic energy, potential energy and velocity for each particle in parallel using dynamic schedule
void kinetic_potential_velocity_dynamic(Particle *particles, int numParticles, int chunk){
    double gravity = 9.81;
    #pragma omp parallel for schedule(dynamic, chunk)
    for (int i = 0; i < numParticles; i++){
        particles[i].velocity = velocity(particles[i].x, particles[i].y, particles[i].time);
        particles[i].kineticEnergy = (particles[i].mass * pow(particles[i].velocity, 2.))/2.;
        particles[i].potentialEnergy = particles[i].mass * gravity * particles[i].y;
    }
}

int main(int argc, char *argv[]) {
    srand(time(NULL)); // Ensures random data each run
	int numParticles, gridSize;
    double pi = 3.1415926536;
    int chunkSize = 10; // Chunk size for Dynamic and Guided schedule, change if needed
    omp_set_num_threads(4); // Number of threads for all parallel regions 
    // Checks if user input is a correct number
    while (1){
        printf("Enter number of particles:\n");
	    scanf("%d", &numParticles);

        if (numParticles <= 0){
            printf("Invalid number of particles, try again\n");
        }else{break;}
    }

    double boxLength, time0 , time1, time2, time3, time4, time5, time6, time7; // Time placeholders and boxLength for particles 
    double areaFrac = 0.3; // area fraction

    if (argc > 1)
    numParticles = atoi(argv[1]);

    boxLength = sqrt(pi*numParticles/areaFrac); // Calculate the boxlength for the simulation
    gridSize = ceil(sqrt(numParticles)); // Assures gridSize * gridSize equals numParticles
    Particle *particles = (Particle *)malloc(numParticles * sizeof(Particle)); // Array of particles

    // Placeholders to store the total energies for each implementation
    double totalKineticEnergySequential, totalPotentialEnergySequential;
    double totalKineticEnergyDynamic, totalPotentialEnergyDynamic;
    double totalKineticEnergyStatic, totalPotentialEnergyStatic;
    double totalKineticEnergyGuided, totalPotentialEnergyGuided;

    // Measures execution time for each function
    initializeParticles(particles, numParticles, boxLength, gridSize); // Initializes the particle array
    time0 = get_walltime ();
    kinetic_potential_velocity_sequential(particles, numParticles);
    time1 = get_walltime ();


    totalKineticEnergySequential = calculateTotalEnergy(particles, numParticles, 'K');
    totalPotentialEnergySequential = calculateTotalEnergy(particles, numParticles, 'P');

    time2 = get_walltime ();
    kinetic_potential_velocity_static(particles, numParticles);
    time3 = get_walltime ();

    totalKineticEnergyStatic = calculateTotalEnergy(particles, numParticles, 'K');
    totalPotentialEnergyStatic = calculateTotalEnergy(particles, numParticles, 'P');

    time4 = get_walltime ();
    kinetic_potential_velocity_dynamic(particles, numParticles, chunkSize);
    time5 = get_walltime ();

    totalKineticEnergyDynamic = calculateTotalEnergy(particles, numParticles, 'K');
    totalPotentialEnergyDynamic = calculateTotalEnergy(particles, numParticles, 'P');

    time6 = get_walltime ();
    kinetic_potential_velocity_guided(particles, numParticles, chunkSize);
    time7 = get_walltime ();

    totalKineticEnergyGuided = calculateTotalEnergy(particles, numParticles, 'K');
    totalPotentialEnergyGuided = calculateTotalEnergy(particles, numParticles, 'P');

    printf("Elapsed time for sequential: %f\n", time1-time0);
    printf("Elapsed time for static schedule: %f\n", time3-time2);
    printf("Elapsed time for dynamic schedule: %f\n", time5-time4);
    printf("Elapsed time for guided schedule: %f\n", time7-time6);
    printf("Kinetic Energy of all implementations...\n");
    equalEnergies(totalKineticEnergySequential, totalKineticEnergyStatic, totalKineticEnergyDynamic, totalKineticEnergyGuided);
    printf("Potential Energy of all implementations...\n");
    equalEnergies(totalPotentialEnergySequential, totalPotentialEnergyStatic, totalPotentialEnergyDynamic, totalPotentialEnergyGuided);
    // Prints the velocity, kinetic and potential energy of the particles, uncomment to use
        // for (int i = 0; i < numParticles; i++) {
        //     printf("Particle %d Kinetic Energy: %f, Potential Energy: %f, and velocity: %f\n", i, particles[i].kineticEnergy, particles[i].potentialEnergy, particles[i].velocity);
        // }
        free(particles);
        return 0; 
    }