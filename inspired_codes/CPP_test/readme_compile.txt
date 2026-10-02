g++ test_surrogate.cpp -o ginn_test -I /home/su_estupinan@private.list.lu/Programs/libtorch/include -I /home/su_estupinan@private.list.lu/Programs/libtorch/include/torch/csrc/api/include -L /home/su_estupinan@private.list.lu/Programs/libtorch/lib -ltorch -ltorch_cpu -lc10 -Wl,-rpath,~/libtorch/lib -std=c++17


Place in same folder:
overlap_model.pt
training_data.csv
scalers.dat

export LD_LIBRARY_PATH=/home/su_estupinan@private.list.lu/Programs/libtorch/lib:$LD_LIBRARY_PATH
./ginn_test
Output:
predictions.csv








g++ test_surrogate_VERTEX.cpp -o ginn_test_VERTEX -I /home/su_estupinan@private.list.lu/Programs/libtorch/include -I /home/su_estupinan@private.list.lu/Programs/libtorch/include/torch/csrc/api/include -L /home/su_estupinan@private.list.lu/Programs/libtorch/lib -ltorch -ltorch_cpu -lc10 -Wl,-rpath,~/libtorch/lib -std=c++17



g++ test_surrogate_VERTEX_overlapOnly.cpp -o test_surrogate_VERTEX_overlapOnly -I /home/su_estupinan@private.list.lu/Programs/libtorch/include -I /home/su_estupinan@private.list.lu/Programs/libtorch/include/torch/csrc/api/include -L /home/su_estupinan@private.list.lu/Programs/libtorch/lib -ltorch -ltorch_cpu -lc10 -Wl,-rpath,~/libtorch/lib -std=c++17
