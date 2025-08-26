// cpp code and PYBIND11_MODULE in the same file

#include <pybind11/pybind11.h>

namespace py = pybind11;

// Tiny C++ class
class Counter {
public:
    Counter(int start) : value(start) {}

    void increment(int delta = 1) { value += delta; }
    int get() const { return value; }

private:
    int value;
};

// Pybind11 binding
PYBIND11_MODULE(example_cpp, m) {
    py::class_<Counter>(m, "Counter")
        .def(py::init<int>(), py::arg("start"))
        .def("increment", &Counter::increment, py::arg("delta") = 1)
        .def("get", &Counter::get);
}
