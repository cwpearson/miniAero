cmake_minimum_required(VERSION 3.16 FATAL_ERROR) # same as Kokkos 4.3

option(MINIAERO_ENABLE_MPI "Enable MPI support" ON)
option(MINIAERO_EXTERNAL_KOKKOS "Use out-of-tree Kokkos" OFF)
option(MINIAERO_KOKKOS_REDUCE_MINMAX_HOST "Run minmax parallel_reduce in host space" OFF)
option(MINIAERO_KOKKOS_BINSORT_HOST "Run Kokkos::Binsort in host space" OFF)
option(MINIAERO_SPLIT_COMPUTE_TANGENTBC_FLUX "Split compute_tangentBC_flux kernel into two smaller kernels" OFF)

project(miniFE LANGUAGES CXX)

# needed for View::dimension_0
if (MINIAERO_EXTERNAL_KOKKOS)
  find_package(Kokkos REQUIRED)
else()
  include(FetchContent)
  message(STATUS "Fetch and configure Kokkos 4.3.00...")
  FetchContent_Declare(Kokkos
    GIT_REPOSITORY    https://github.com/kokkos/kokkos.git
    GIT_TAG           4.3.00
  )
  FetchContent_MakeAvailable(Kokkos)
endif()

if(MINIAERO_ENABLE_MPI)
  find_package(MPI REQUIRED)
else()
  find_package(MPI) # okay if we can't find MPI if the user didn't ask for it
endif()

file(GLOB MINIAERO_SRCS "*.C")
set(MINIAERO_INCL_DIRS ${CMAKE_CURRENT_LIST_DIR})

add_executable(miniAero.kokkos ${MINIAERO_SRCS})
target_link_libraries(miniAero.kokkos Kokkos::kokkos)
foreach(INCL_DIR ${MINIAERO_INCL_DIRS})
  target_include_directories(miniAero.kokkos PRIVATE ${INCL_DIR})
endforeach()
target_compile_definitions(miniAero.kokkos PRIVATE ATOMICS_FLUX) # needed for correctness
if (MINIAERO_KOKKOS_REDUCE_MINMAX_HOST)
  target_compile_definitions(miniAero.kokkos PRIVATE MINIAERO_KOKKOS_REDUCE_MINMAX_HOST)
endif()
if (MINIAERO_KOKKOS_BINSORT_HOST)
  target_compile_definitions(miniAero.kokkos PRIVATE MINIAERO_KOKKOS_BINSORT_HOST)
endif()
if (MINIAERO_SPLIT_COMPUTE_TANGENTBC_FLUX)
  target_compile_definitions(miniAero.kokkos PRIVATE MINIAERO_SPLIT_COMPUTE_TANGENTBC_FLUX)
endif()

if (MPI_FOUND AND MINIAERO_ENABLE_MPI)
  message(STATUS "MPI_VERSION =                 ${MPI_VERSION}")
  message(STATUS "MPI_CXX_COMPILER =            ${MPI_CXX_COMPILER}")
  message(STATUS "MPI_CXX_COMPILE_OPTIONS =     ${MPI_CXX_COMPILE_OPTIONS}")
  message(STATUS "MPI_CXX_COMPILE_DEFINITIONS = ${MPI_CXX_COMPILE_DEFINITIONS}")
  message(STATUS "MPI_CXX_INCLUDE_DIRS =        ${MPI_CXX_INCLUDE_DIRS}")
  message(STATUS "MPI_CXX_LINK_FLAGS =          ${MPI_CXX_LINK_FLAGS}")
  message(STATUS "MPI_CXX_LIBRARIES =           ${MPI_CXX_LIBRARIES}")
  target_compile_definitions(miniAero.kokkos PRIVATE WITH_MPI=1)
  target_link_libraries(miniAero.kokkos MPI::MPI_CXX)
endif()

enable_testing()


if (MPI_FOUND AND MINIAERO_ENABLE_MPI)
  add_test(
      NAME AllTests-MPI
      WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/tests
      COMMAND run_tests.sh ${CMAKE_CURRENT_BINARY_DIR}/miniAero.kokkos MPI
  )
else()
  add_test(
      NAME AllTests 
      WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/tests
      COMMAND run_tests.sh ${CMAKE_CURRENT_BINARY_DIR}/miniAero.kokkos
  )
endif()