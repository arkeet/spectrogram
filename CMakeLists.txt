cmake_minimum_required(VERSION 2.8)
project(spectrogram C)

find_package(OpenGL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW REQUIRED fftw3)
pkg_search_module(GLFW REQUIRED libglfw)
pkg_search_module(PORTAUDIO REQUIRED portaudio-2.0)

add_executable(spectrogram main)
target_link_libraries(spectrogram
    ${OPENGL_LIBRARIES}
    ${FFTW_LIBRARIES}
    ${GLFW_LIBRARIES}
    ${PORTAUDIO_LIBRARIES}
    )
