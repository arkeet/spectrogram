OUT=spectrogram
CC=gcc -O2
PKGS=pangocairo portaudio-2.0 libglfw fftw3
CFLAGS=$(shell pkg-config --cflags $(PKGS))
LIBS=$(shell pkg-config --libs $(PKGS)) -lGLU

all: $(OUT)

$(OUT): main.o
	$(CC) -o $@ $^ $(LIBS)

%.o: %.c
	$(CC) -o $@ $(CFLAGS) -c $<

clean:
	rm -f $(OUT) *.o

.PHONY: clean
