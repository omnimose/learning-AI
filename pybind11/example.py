"""
1. Need to run pip install ninja

2. Start VS Code from the VC tools cmd prompt so it can find the cl.exe tool chain:

C:\Program Files\Microsoft Visual Studio\2022-1\Enterprise\VC\Auxiliary\Build>vcvars64.bat
**********************************************************************
** Visual Studio 2022 Developer Command Prompt v17.14.12
** Copyright (c) 2025 Microsoft Corporation
**********************************************************************
[vcvarsall.bat] Environment initialized for: 'x64'

C:\Program Files\Microsoft Visual Studio\2022-1\Enterprise\VC\Auxiliary\Build>

(.venv) PS C:\Users\georgel\src\learning-AI>  & 'c:\Users\georgel\src\learning-AI\.venv\Scripts\python.exe' 'c:\Users\georgel\.vscode\extensions\ms-python.debugpy-2025.10.0-win32-x64\bundled\libs\debugpy\launcher' '64972' '--' 'c:\Users\georgel\src\learning-AI\pybind11\example.py' 
Using C:\Users\georgel\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\Local\torch_extensions\torch_extensions\Cache\py311_cpu as PyTorch extensions root...
Emitting ninja build file C:\Users\georgel\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\Local\torch_extensions\torch_extensions\Cache\py311_cpu\example_cpp\build.ninja...
Building extension module example_cpp...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
[1/2] cl /showIncludes -DTORCH_EXTENSION_NAME=example_cpp -DTORCH_API_INCLUDE_EXTENSION_H -Ic:\Users\georgel\src\learning-AI\.venv\Lib\site-packages\torch\include -Ic:\Users\georgel\src\learning-AI\.venv\Lib\site-packages\torch\include\torch\csrc\api\include -Ic:\Users\georgel\src\learning-AI\.venv\Lib\site-packages\torch\include\TH -Ic:\Users\georgel\src\learning-AI\.venv\Lib\site-packages\torch\include\THC "-IC:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.11_3.11.2544.0_x64__qbz5n2kfra8p0\Include" -D_GLIBCXX_USE_CXX11_ABI=0 /MD /wd4819 /wd4251 /wd4244 /wd4267 /wd4275 /wd4018 /wd4190 /wd4624 /wd4067 /wd4068 /EHsc /std:c++17 -O3 -std=c++17 -c c:\Users\georgel\src\learning-AI\pybind11\example.cpp /Foexample.o
Microsoft (R) C/C++ Optimizing Compiler Version 19.44.35214 for x64
Copyright (C) Microsoft Corporation.  All rights reserved.

cl : Command line warning D9002 : ignoring unknown option '-O3'
cl : Command line warning D9002 : ignoring unknown option '-std=c++17'
[2/2] "C:\Program Files\Microsoft Visual Studio\2022-1\Enterprise\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64/link.exe" example.o /nologo /DLL c10.lib torch_cpu.lib torch.lib /LIBPATH:c:\Users\georgel\src\learning-AI\.venv\Lib\site-packages\torch\lib torch_python.lib "/LIBPATH:C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.11_3.11.2544.0_x64__qbz5n2kfra8p0\libs" /out:example_cpp.pyd
   Creating library example_cpp.lib and object example_cpp.exp
Loading extension module example_cpp...

3. The generate Python module is at PyTorch extensions root:
C:\Users\georgel\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\Local\torch_extensions\torch_extensions\Cache\py311_cpu 
"""

import os
from torch.utils.cpp_extension import load

# Get current directory
_this_dir = os.path.dirname(os.path.abspath(__file__))

# Compile and load the C++ extension
example_cpp = load(
    name="example_cpp",
    sources=[os.path.join(_this_dir, "example.cpp")],
    extra_cflags=["-O3", "-std=c++17"],  # C++17 is enough
    is_python_module=True,  # <-- generates example_cpp.pyd in current dir, this is ChatGPT suggestion, seems it is wrong.
    verbose=True
)

# it is automatically loaded 
# so we can use it here

# Create an instance
c = example_cpp.Counter(10)

# Use methods
print("Initial value:", c.get())  # 10
c.increment()
print("After increment:", c.get())  # 11
c.increment(5)
print("After increment by 5:", c.get())  # 16
