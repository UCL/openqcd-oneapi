#!/usr/bin/env python3
from sys import argv, stdout, exit
import os
from subprocess import call


'''
Steps:
1. A file, e.g. 'options_to_test.txt' is read with one whole set of option per line. 
Each line means one test. 
Comments with '#' and white lines are ignored.

2. A second file 'proto/compile_settings' with this structure is read and 
a modified version is printed out.  Notice '{}' next to CFLAGS - it's crucial.

##### EXAMPLE #####
CODELOC /work/dp006/dp006/dc-mesi1/OpenQCDCode
MPI_HOME /bgsys/drivers/ppcfloor/comm/xl
MPI_INCLUDE /bgsys/drivers/ppcfloor/comm/xl/include
GCC  /usr/bin/gcc
CFLAGS {}
NPROC0_TOT   4 
NPROC1_TOT   8
NPROC2_TOT   8 
NPROC3_TOT   8 
L0 8       
L1 4       
L2 4       
L3 4       
NPROC0_BLK 1
NPROC1_BLK 1
NPROC2_BLK 1
NPROC3_BLK 1
##### EXAMPLE END #####

3. For each test:
    a. A directory is created.
    b. The modified file is written in that directory
    c. other necessary files are copied in.   
'''
try :
    options_file = argv[1]
except:
    options_file = 'options_to_test.txt'

with open(options_file,'r') as f :
    lines = f.readlines()

try :
    basedir = argv[2]
except:
    basedir = 'proto'
    print("Proto dir not given, using 'proto'")

with open(basedir + '/compile_settings.txt', 'r') as f:
    compile_settings_text = f.read()

for line in lines:
    id = line.find('#')
    if id == -1:
        id = len(line)       
    line = line[:id]

    if len(line) > 1 :
        dirname= 'bench' + line.replace(' ','')[:-1]
        if basedir != 'proto':
            dirname = basedir + dirname ; 
    
        try:
            os.makedirs(dirname)
        except OSError :
            print("OSERROR")
    
        for filename in ["Makefile","run.pbs","benchmark.in" ]:
             call(["ln", "-s", "../"+ basedir+'/'+filename , dirname + "/"+ filename ])
        
        with open(dirname+'/compile_settings.txt', 'w') as fout:
            fout.write(compile_settings_text.format(line))
        
        print(dirname)






