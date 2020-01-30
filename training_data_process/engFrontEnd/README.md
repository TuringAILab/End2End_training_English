ctypes.cdll.LoadLibrary('/abs/path/to/a.so')
ctypes.cdll.LoadLibrary('/abs/path/to/b.so')

I get an error on the second load, because b.so refers to simply 'a.so', without an rpath, and so b.so doesn't know that's the correct a.so. So I have to set LD_LIBRARY_PATH in advance to include '/abs/path/to'.

To avoid having to set LD_LIBRARY_PATH, you modify the rpath entry in the .so files. On Linux, there are two utilities I found that do this: chrpath, and patchelf. chrpath is available from the Ubuntu repositories. It cannot change rpath on .so's that never had one. patchelf is more flexible.


sudo apt install patchelf 

patchelf --set-rpath ./engFrontEnd libEngTTS.so
patchelf --set-rpath ./engFrontEnd libflite.so
patchelf --set-rpath ./engFrontEnd libhts_engine.so
patchelf --set-rpath ./engFrontEnd libmanager.so
patchelf --set-rpath ./engFrontEnd libtextinternal.so
