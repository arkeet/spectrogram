#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <complex.h>
#include <pthread.h>

#include <fftw3.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glfw.h>
#include <portaudio.h>

#ifndef GL_TEXTURE_RECTANGLE
#define GL_TEXTURE_RECTANGLE 34037
#endif

#define TWOPI (2*M_PI)
#define SQRTWOPI (sqrt(2*M_PI))

#define WIDTH 800
#define HEIGHT 600
#define SAMPLERATE 22050
#define BUFSIZE 512
#define SAMPSIZE 2048
#define NBUFFERS 16

#define IX(arr,stride,x,y) ((arr)[(y)*(stride)+(x)])

#define TABLESIZE 4096
#define TABLERES 512

bool done;
bool paused;
double fps;
int brightness;
int displayperiod;
int winwidth;
bool redraw;

double window[SAMPSIZE];
double window_dt[SAMPSIZE];
double logistic_table[TABLESIZE];

enum {
    gaussian,
    hann,
    nuttall,
    rect,
    MAX_WINDOW_TYPE
} windowfunction;
const char *window_names[] = {
    "Gaussian",
    "Hann",
    "Nuttall",
    "Rectangle",
    0
};

void init_tables()
{
    double scl = 1.0/winwidth;
    int i;
    for (i = 0; i < SAMPSIZE; i++) {
        double z = (double)(i - SAMPSIZE/2) / (winwidth);

        switch (windowfunction) {
            case gaussian:
                z *= 2;
                window[i] = exp(-z*z/2) / SQRTWOPI * scl * 2;
                window_dt[i] = window[i] * -SAMPLERATE*z/winwidth;
                break;
            case hann:
                if (z <= -1 || z >= 1) {
                    window[i] = 0;
                } else {
                    window[i] = (cos(M_PI*z)+1)/2 * scl;
                }
                break;
            case nuttall:
                if (z <= -1 || z >= 1) {
                    window[i] = 0;
                } else {
                    window[i] = 0.355768;
                    window[i] += 0.487396 * cos(M_PI*z);
                    window[i] += 0.144232 * cos(M_PI*2*z);
                    window[i] += 0.012604 * cos(M_PI*3*z);
                    window[i] *= scl;
                }
                break;
            case rect:
                if (z <= -1 || z >= 1) {
                    window[i] = 0;
                } else {
                    window[i] = scl/2;
                }
                break;
        }
    }

    for (i = 0; i < TABLESIZE; i++) {
        double x = (double)(i - TABLESIZE/2) / TABLERES;
        logistic_table[i] = 1 / (1 + exp(-x));
    }
}

int GLFWCALL windowCloseCallback()
{
    done = true;
    return GL_TRUE;
}

void GLFWCALL keyCallback(int key, int action)
{
    switch (key) {
        case GLFW_KEY_ESC:
            done = true;
            break;
        case GLFW_KEY_SPACE:
            if (action == GLFW_PRESS) {
                paused = !paused;
                printf("%s\n", paused ? "paused" : "unpaused");
            }
            break;
        case GLFW_KEY_UP:
        case 'K':
            if (action == GLFW_PRESS) {
                brightness += 5;
                redraw = true;
                printf("Brightness: %d\n", brightness);
            }
            break;
        case GLFW_KEY_DOWN:
        case 'J':
            if (action == GLFW_PRESS) {
                brightness -= 5;
                redraw = true;
                printf("Brightness: %d\n", brightness);
            }
            break;
        case GLFW_KEY_LEFT:
        case 'H':
            if (action == GLFW_PRESS) {
                displayperiod *= 2;
                if (displayperiod > BUFSIZE) {
                    displayperiod = BUFSIZE;
                }
                printf("Period: %d\n", displayperiod);
            }
            break;
        case GLFW_KEY_RIGHT:
        case 'L':
            if (action == GLFW_PRESS) {
                displayperiod /= 2;
                if (displayperiod < 1) {
                    displayperiod = 1;
                }
                printf("Period: %d\n", displayperiod);
            }
            break;
        case '=':
            if (action == GLFW_PRESS) {
                winwidth *= 2;
                if (winwidth > SAMPSIZE/2) {
                    winwidth = SAMPSIZE/2;
                }
                init_tables();
                printf("Window width: %d\n", winwidth);
            }
            break;
        case '-':
            if (action == GLFW_PRESS) {
                winwidth /= 2;
                if (winwidth < 1) {
                    winwidth = 1;
                }
                init_tables();
                printf("Window width: %d\n", winwidth);
            }
            break;
        case '0':
            if (action == GLFW_PRESS) {
                windowfunction++;
                if (windowfunction >= MAX_WINDOW_TYPE) {
                    windowfunction = 0;
                }
                init_tables();
                printf("Window: %s\n", window_names[windowfunction]);
            }
            break;
    }
}

double logistic(double x)
{
    int i = x * TABLERES + TABLESIZE/2;
    if (i < 0) return 0;
    if (i >= TABLESIZE) return 1;
    return logistic_table[i];
}

double log_fast(double a) {
    // Thanks Edward Kmett.
    union { double d; long long x; } u = { a };
    return (u.x - 4606921278410026770) * 1.539095918623324e-16; // 1 / 6497320848556798.0;
}

double log10_fast(double a) {
    union { double d; long long x; } u = { a };
    return (u.x - 4606921278410026770) * 6.684208645779258e-17;
}

uint32_t colourmap(double x)
{
    uint8_t rgba[4];
    double logx = log10_fast(x) + (double)brightness/10;
    rgba[0] = logistic(logx-0) * 255;
    rgba[1] = logistic(logx-2) * 255;
    rgba[2] = logistic(logx-4) * 255;
    rgba[3] = 255;
    return *(uint32_t*)rgba;
}

double gettime()
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1.0e-9;
}

pthread_mutex_t mutex;

struct mydata {
    int time;
    int nextbuf;
    double *buffers;
};

static int paCallback(const void *inputBuffer, void *outputBuffer,
        unsigned long framesPerBuffer,
        const PaStreamCallbackTimeInfo* timeInfo,
        PaStreamCallbackFlags statusFlags, void *userData)
{
    const float* in = inputBuffer;
    struct mydata *mydata = userData;
    static int nextbuf = 0;
    double* buf = mydata->buffers + BUFSIZE * nextbuf;

    pthread_mutex_lock(&mutex);
    mydata->time += BUFSIZE;
    int i;
    for (i = 0; i < BUFSIZE; i++) {
        buf[i] = in[i];
    }
    nextbuf++;
    if (nextbuf == NBUFFERS) {
        nextbuf = 0;
    }
    mydata->nextbuf = nextbuf;
    pthread_mutex_unlock(&mutex);

    return 0;
}

float scale(float l)
{
    return -(l - 1)/13;
}

int main(int argc, char* argv[])
{
    glfwInit();
    glfwOpenWindowHint(GLFW_WINDOW_NO_RESIZE, GL_TRUE);
    glfwOpenWindow(WIDTH, HEIGHT, 0, 0, 0, 0, 0, 0, GLFW_WINDOW);
    glfwSetWindowTitle("spectrogram");
    glfwSetWindowCloseCallback(windowCloseCallback);
    glfwSetKeyCallback(keyCallback);

    glEnable(GL_TEXTURE_RECTANGLE);

    glViewport(0, 0, WIDTH, HEIGHT);

    glMatrixMode(GL_PROJECTION);
    gluOrtho2D(0, WIDTH, HEIGHT, 0);

    glMatrixMode(GL_MODELVIEW);

    Pa_Initialize();

    done = false;
    paused = false;
    brightness = 50;
    displayperiod = 128;
    winwidth = 256;

    const int FTSIZE = SAMPSIZE/2 + 1;
    double *screen_fl = calloc(WIDTH*FTSIZE, sizeof(double));
    uint32_t *screen = calloc(WIDTH*FTSIZE, sizeof(uint32_t));
    memset(screen, 0, sizeof(uint32_t)*WIDTH*FTSIZE);

    glEnable(GL_TEXTURE_RECTANGLE);

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_RECTANGLE, tex);
    glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA, FTSIZE, WIDTH, 0,
            GL_RGBA, GL_UNSIGNED_BYTE, 0);

    double buf[BUFSIZE*NBUFFERS];
    double ft_in[SAMPSIZE];
    fftw_complex ft[FTSIZE];
    fftw_complex ft_dt[FTSIZE];

    memset(buf, 0, sizeof(buf));

    pthread_mutex_init(&mutex, NULL);

    struct mydata mydata;
    mydata.time = 0;
    mydata.nextbuf = 0;
    mydata.buffers = buf;

    int nextbuf = 0;

    fftw_plan plan = fftw_plan_dft_r2c_1d(SAMPSIZE, ft_in, ft, FFTW_MEASURE);
    fftw_plan plan_dt = fftw_plan_dft_r2c_1d(SAMPSIZE, ft_in, ft_dt, FFTW_MEASURE);

    PaStream* stream;
    Pa_OpenDefaultStream(&stream, 1, 0, paFloat32, SAMPLERATE, BUFSIZE,
            paCallback, &mydata);
    Pa_StartStream(stream);

    int frame_num = 0;
    double update_time = gettime();
    double frame_time = gettime();
    fps = 1;

    int buftime = 0;

    init_tables();

    int x, y;
    x = 0;

    while (!done) {
        int i, j, k;

        int x_min = WIDTH;
        int x_max = -1;

        double processtime;

        int peakid;
        int nloop = 0;
        while (!paused && mydata.time > buftime) {
            pthread_mutex_lock(&mutex);

            if (mydata.time - buftime > BUFSIZE * NBUFFERS || nloop++ > 7) {
                printf("dropped\n", mydata.time);
                buftime = mydata.time;
                nextbuf = mydata.nextbuf;
                pthread_mutex_unlock(&mutex);
                break;
            }

            processtime = gettime();
            for (j = 0; j < BUFSIZE; j += displayperiod) {
                double *curbuf;

                curbuf = buf + (nextbuf + NBUFFERS) * BUFSIZE - SAMPSIZE + j;
                for (i = 0; i < SAMPSIZE; i++) {
                    if (curbuf > buf + NBUFFERS * BUFSIZE) {
                        curbuf -= NBUFFERS * BUFSIZE;
                    }
                    ft_in[i] = window[i] * *curbuf;
                    curbuf++;
                }
                fftw_execute(plan);

                curbuf = buf + (nextbuf + NBUFFERS) * BUFSIZE - SAMPSIZE + j;
                for (i = 0; i < SAMPSIZE; i++) {
                    if (curbuf > buf + NBUFFERS * BUFSIZE) {
                        curbuf -= NBUFFERS * BUFSIZE;
                    }
                    ft_in[i] = window_dt[i] * *curbuf;
                    curbuf++;
                }
                fftw_execute(plan_dt);

                for (k = 0; k < FTSIZE; k++) {
                    y = k;
                    double absft = cabs(ft[k]);
                    IX(screen_fl, FTSIZE, y, x) = absft * absft;
                }

                if (x < x_min) x_min = x;
                if (x > x_max) x_max = x;
                x = (x + 1) % WIDTH;
            }
            processtime = gettime() - processtime;

            buftime += BUFSIZE;
            nextbuf = (nextbuf + 1) % NBUFFERS;

            pthread_mutex_unlock(&mutex);
        }

        if (redraw || !paused && nloop > 0) {
            if (redraw) {
                x_min = 0;
                x_max = WIDTH - 1;
                redraw = false;
            }
            for (i = x_min; i <= x_max; i++) {
                for (j = 0; j < FTSIZE; j++) {
                    IX(screen, FTSIZE, j, i) = colourmap(IX(screen_fl, FTSIZE, j, i));
                }
            }
            glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA, FTSIZE, WIDTH, 0,
                    GL_RGBA, GL_UNSIGNED_BYTE, screen);
        }

        if (!paused && nloop == 0) {
            continue;
        }

        glClearColor(1, 1, 1, 0);
        glClear(GL_COLOR_BUFFER_BIT);

        glBegin(GL_QUADS);
        glTexCoord2f(FTSIZE, 0);
        glVertex2f(0, 0);
        glTexCoord2f(FTSIZE, WIDTH);
        glVertex2f(WIDTH, 0);
        glTexCoord2f(0, WIDTH);
        glVertex2f(WIDTH, HEIGHT);
        glTexCoord2f(0, 0);
        glVertex2f(0, HEIGHT);
        glEnd();

        glfwSwapBuffers();

        double new_time = gettime();
        double dt = new_time - frame_time;
        frame_time = new_time;
        fps = fps + (1/dt - fps) * 0.05;

        if (new_time - update_time >= 0.1) {
            char title[128];
            snprintf(title, 128, "spectrogram - %.3f fps - %.2f ms\n", fps,
                    processtime * 1000);
            glfwSetWindowTitle(title);
            update_time = new_time;
        }
    }

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();
    glfwTerminate();

    free(screen_fl);
    free(screen);

    return 0;
}
