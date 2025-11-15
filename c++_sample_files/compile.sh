mkdir build
cd build
cmake .. -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) \
 -DCMAKE_C_COMPILER=gcc \
 -DCMAKE_CXX_COMPILER=g++
cd build
make 
cd ..