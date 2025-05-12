cd gem5-aladdin && git pull origin master && git submodule update --init --recursive && cd ..
cd LLVM-Tracer && git pull origin master && cd ..
cd smaug && git pull origin master && git submodule update --init --recursive && cd ..
cd /workspace/gem5-aladdin
python2.7 `which scons` build/X86/gem5.opt PROTOCOL=MESI_Two_Level_aladdin -j2
cd /workspace/smaug
make all -j8